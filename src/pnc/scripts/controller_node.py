#!/usr/bin/env python3

import rospy
import numpy as np
import math
from enum import Enum
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
import tf.transformations as tft
from NMPC_controller import AutonomousCarNMPC  

# Define finite state machine states
class FSMState(Enum):
    IDLE = 0
    ROTATE_TO_TRAJECTORY = 1  # Rotate to align with the trajectory's starting orientation
    MPC_TRACKING = 2          # Perform NMPC tracking
    ROTATE_TO_DEFAULT = 3     # Final rotation to a default key orientation

# Simple PID controller for yaw alignment
class PIDController:
    def __init__(self, kp, ki, kd, max_output):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        # Saturate the output
        return max(min(output, self.max_output), -self.max_output)

# Convert quaternion to yaw angle
def quat_to_yaw(q):
    euler = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
    return euler[2]

# Convert a Pose message to a numpy array [x, y, yaw]
def pose_to_np(pose):
    x = pose.position.x
    y = pose.position.y
    yaw = quat_to_yaw(pose.orientation)
    return np.array([x, y, yaw])

# NMPC Path Tracker Node using only /local_path for path information (end_path)
class NMPCPathTracker:
    def __init__(self):
        rospy.init_node("nmpc_path_tracker")

        # Load parameters from the parameter server
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.odom_topic = rospy.get_param("~odom_topic", "/odometry/filtered")
        self.vel_ref = rospy.get_param("~vel_ref", 1.5)
        self.xy_tol = rospy.get_param("~xy_tol", 0.1)
        self.rot_tol = np.deg2rad(rospy.get_param("~rot_tol_deg", 10))
        self.rate_hz = rospy.get_param("~rate", 10)
        self.initial_state = np.array(rospy.get_param("~initial_state", [0.0, 0.0, 0.0]))

        self.max_lin_acc = rospy.get_param("~max_lin_acc", 1.0)
        self.max_ang_acc = rospy.get_param("~max_ang_acc", 0.5)
        self.max_lin_vel = rospy.get_param("~max_lin_vel", 1.0)
        self.max_ang_vel = rospy.get_param("~max_ang_vel", 1.0)
        self.vel_penalty = rospy.get_param("~vel_penalty", 0.0)

        # NMPC parameters
        self.N = rospy.get_param("~N", 30)
        self.T = rospy.get_param("~T", 0.1)
        Q_list = rospy.get_param("~Q", [3, 3, 1])
        R_list = rospy.get_param("~R", [0.2, 0.2])
        self.Q = np.diag(Q_list)
        self.R = np.diag(R_list)

        self.use_kalman_filter = rospy.get_param("~use_kalman_filter", True)
        self.kalman_Q = np.eye(3) * 1.0# Trusting the model more 0.8
        self.kalman_R = np.eye(3) * 0.1# Reducing the weight of noisy sensor data

 
        params = {
            "Q": self.Q,
            "R": self.R,
            "initial_state": self.initial_state,
            "max_lin_acc": self.max_lin_acc,
            "max_ang_acc": self.max_ang_acc,
            "max_lin_vel": self.max_lin_vel,
            "max_ang_vel": self.max_ang_vel,
            "vel_ref": self.vel_ref,
            "vel_penalty": self.vel_penalty, 
            "kalman_Q": self.kalman_Q,
            "kalman_R" : self.kalman_R,
            "use_kalman_filter": self.use_kalman_filter,
        }
        # Initialize the NMPC controller with the given parameters
        self.nmpc_controller = AutonomousCarNMPC(T=self.T, N=self.N, param=params)

        # Initialize PID controllers for yaw and position control
        self.yaw_pid = PIDController(kp=2.0, ki=0.0, kd=0.1, max_output=1.0)
        self.pos_pid = PIDController(kp=1.0, ki=0.0, kd=0.1, max_output=1.0)

        # Initialize state variables
        self.current_odom = None
        self.current_pose = None   # Numpy array: [x, y, yaw]
        self.end_path = None       # Combined global+local path from /local_path topic
        self.fsm_state = FSMState.IDLE

        # ROS publishers and subscribers
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        # self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)
        self.amcl_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.amcl_pose_callback)
        self.end_path_sub = rospy.Subscriber("/local_path", Path, self.end_path_callback)

        rospy.loginfo("NMPC Path Tracker node initialized using /local_path for path (end_path).")

    # Odometry callback: update current pose from odometry messages
    def odom_callback(self, msg):
        self.current_odom = msg
        self.current_pose = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            quat_to_yaw(msg.pose.pose.orientation)
        ])

    def amcl_pose_callback(self, msg):
        self.current_pose = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            quat_to_yaw(msg.pose.pose.orientation)
        ])

    # /local_path callback: store the received path as end_path
    # When a valid end_path is received, and if the robot is in IDLE (i.e., stationary), switch to the next stage
    def end_path_callback(self, msg):
        self.end_path = msg
        rospy.loginfo("Received /local_path with %d poses.", len(msg.poses))
        if self.fsm_state == FSMState.IDLE:
            rospy.loginfo("Switching to ROTATE_TO_TRAJECTORY state based on end_path reception.")
            self.fsm_state = FSMState.ROTATE_TO_TRAJECTORY

    # Extract a local segment (N+1 poses) from the end_path
    def get_local_ref_path(self):
        if self.end_path is None or self.current_pose is None:
            return None

        poses = self.end_path.poses
        # Find the index of the pose closest to the current robot state
        min_dist = float("inf")
        nearest_idx = 0
        for i, pose_st in enumerate(poses):
            p = pose_to_np(pose_st.pose)
            dist = np.linalg.norm(p[:2] - self.current_pose[:2])
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        # Extract the next N+1 poses; pad with the last pose if necessary
        local_poses = poses[nearest_idx:nearest_idx + self.N + 1]
        if len(local_poses) < self.N + 1:
            while len(local_poses) < self.N + 1:
                local_poses.append(local_poses[-1])
        ref_path = [pose_to_np(ps.pose) for ps in local_poses]
        return np.array(ref_path)

    # Compute the control command using the NMPC controller based on the reference trajectory
    def compute_cmd_vel(self, ref_path):
        if ref_path is None or self.current_pose is None:
            return None
        
        current_u = (self.nmpc_controller.U_opt[0, :]
                     if self.nmpc_controller.U_opt.size
                     else np.zeros(self.nmpc_controller.control_dim))
        # Update state estimate using the Kalman filter
        if self.use_kalman_filter:
            current_state = self.nmpc_controller.kalman_update(current_u, self.current_pose)
        else:
            current_state = self.current_pose
        # Solve the NMPC problem with the current state and reference trajectory
        u = self.nmpc_controller.solve_nmpc(current_state, ref_path, current_u)

        cmd = Twist()
        cmd.linear.x = u[0]
        cmd.angular.z = u[1]
        return cmd

    # Main loop: execute state machine logic
    def run(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            if self.current_pose is None:
                rospy.logwarn("Current pose is None.")
                rate.sleep()
                continue

            # Debug: print current FSM state
            rospy.loginfo("Current FSM state: %s", self.fsm_state.name)

            if self.fsm_state == FSMState.ROTATE_TO_TRAJECTORY:
                # Rotate to align with the trajectory's starting orientation
                ref_path = self.get_local_ref_path()
                if ref_path is None:
                    rospy.logwarn("Reference path is None in ROTATE_TO_TRAJECTORY state.")
                    rate.sleep()
                    continue
                desired_yaw = math.atan2(ref_path[1, 1] - ref_path[0, 1],
                                         ref_path[1, 0] - ref_path[0, 0])
                yaw_error = desired_yaw - self.current_pose[2]
                yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
                rospy.loginfo("Desired yaw: %.2f, Current yaw: %.2f, Yaw error: %.2f", desired_yaw, self.current_pose[2], yaw_error)
                if abs(yaw_error) < self.rot_tol:
                    rospy.loginfo("Rotation to trajectory orientation completed.")
                    self.fsm_state = FSMState.MPC_TRACKING
                else:
                    twist = Twist()
                    twist.linear.x = 0
                    twist.angular.z = self.yaw_pid.compute(yaw_error, 1.0 / self.rate_hz)
                    rospy.loginfo("Publishing rotation cmd_vel: linear.x=%.2f, angular.z=%.2f", twist.linear.x, twist.angular.z)
                    self.cmd_vel_pub.publish(twist)
                    rate.sleep()
                    continue

            elif self.fsm_state == FSMState.MPC_TRACKING:
                # Use NMPC to track the reference trajectory
                ref_path = self.get_local_ref_path()
                if ref_path is None:
                    rospy.logwarn("Reference path is None in MPC_TRACKING state.")
                    rate.sleep()
                    continue
                cmd = self.compute_cmd_vel(ref_path)
                if cmd is not None:
                    rospy.loginfo("Publishing MPC cmd_vel: linear.x=%.2f, angular.z=%.2f", cmd.linear.x, cmd.angular.z)
                    self.cmd_vel_pub.publish(cmd)
                else:
                    rospy.logwarn("Computed cmd is None in MPC_TRACKING state.")
                # For this version, no additional condition is required to switch state.
                # You can add further logic here if needed.

            elif self.fsm_state == FSMState.ROTATE_TO_DEFAULT:
                # Rotate to the default final orientation (specified by a ROS parameter)
                default_yaw = rospy.get_param("~default_final_yaw", 0.0)
                yaw_error = default_yaw - self.current_pose[2]
                yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
                rospy.loginfo("Final rotation: default_yaw=%.2f, current_yaw=%.2f, yaw_error=%.2f", default_yaw, self.current_pose[2], yaw_error)
                if abs(yaw_error) < self.rot_tol:
                    rospy.loginfo("Final default rotation completed. Vehicle is now aligned.")
                    self.cmd_vel_pub.publish(Twist())  # Stop the vehicle
                    self.fsm_state = FSMState.IDLE
                else:
                    twist = Twist()
                    twist.linear.x = 0
                    twist.angular.z = self.yaw_pid.compute(yaw_error, 1.0 / self.rate_hz)
                    rospy.loginfo("Publishing final rotation cmd_vel: linear.x=%.2f, angular.z=%.2f", twist.linear.x, twist.angular.z)
                    self.cmd_vel_pub.publish(twist)

            elif self.fsm_state == FSMState.IDLE:
                # In IDLE state, stop the robot
                self.cmd_vel_pub.publish(Twist())

            rate.sleep()

if __name__ == "__main__":
    try:
        tracker = NMPCPathTracker()
        tracker.run()
    except rospy.ROSInterruptException:
        pass
