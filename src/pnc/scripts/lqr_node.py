#!/usr/bin/env python3

import rospy
import numpy as np
import math
from enum import Enum
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
import tf.transformations as tft
import scipy.linalg

# Define finite state machine states
class FSMState(Enum):
    IDLE = 0
    ROTATE_TO_TRAJECTORY = 1  # Rotate to align with the trajectory's starting direction
    LQR_TRACKING = 2          # Execute LQR tracking
    ROTATE_TO_DEFAULT = 3     # Rotate to the default final orientation

# PID controller for heading alignment during state transitions
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

# LQR Path Tracking Node using the error state [e_x, e_theta]
# The controller directly computes control commands (linear velocity and angular velocity)
# without adding a reference speed.
class LQRPathTracker:
    def __init__(self):
        rospy.init_node("lqr_path_tracker")

        # Load parameters from the parameter server
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.xy_tol = rospy.get_param("~xy_tol", 0.1)
        self.rot_tol = np.deg2rad(rospy.get_param("~rot_tol_deg", 10))
        self.rate_hz = rospy.get_param("~rate", 20)
        self.initial_state = np.array(rospy.get_param("~initial_state", [0.0, 0.0, 0.0]))

        self.max_lin_vel = rospy.get_param("~max_lin_vel", 1.0)
        self.max_ang_vel = rospy.get_param("~max_ang_vel", 1.0)

        # LQR parameters: Q for state error and R for control input
        # We design the controller based on the state [e_x, e_theta]
        Q_list = rospy.get_param("~Q", [5, 1])  # weights for e_x and e_theta
        R_list = rospy.get_param("~R", [0.2, 0.2])
        self.Q = np.diag(Q_list)
        self.R = np.diag(R_list)

        # State-space model for the error dynamics (only e_x and e_theta)
        # Dynamics: e_dot_x = delta_v,  e_dot_theta = delta_omega
        # Thus, A = [[0, 0], [0, 0]] and B = identity (2x2)
        A = np.array([[0, 0],
                      [0, 0]])
        B = np.array([[1, 0],
                      [0, 1]])
        # Solve the continuous-time algebraic Riccati equation (CARE)
        P = scipy.linalg.solve_continuous_are(A, B, self.Q, self.R)
        # Compute the LQR gain matrix K
        self.lqr_K = np.linalg.inv(self.R) @ B.T @ P

        # Initialize the PID controller for heading alignment during rotation
        self.yaw_pid = PIDController(kp=2.0, ki=0.0, kd=0.1, max_output=1.0)

        # Initialize state variables
        self.current_pose = None   # Current pose as [x, y, theta]
        self.end_path = None       # Received path from /local_path topic
        self.fsm_state = FSMState.IDLE

        # ROS publishers and subscribers
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.amcl_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.amcl_pose_callback)
        self.end_path_sub = rospy.Subscriber("/local_path", Path, self.end_path_callback)

        rospy.loginfo("LQR Path Tracker node started, using /local_path for path information.")

    # Callback for pose messages from AMCL
    def amcl_pose_callback(self, msg):
        self.current_pose = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            quat_to_yaw(msg.pose.pose.orientation)
        ])

    # Callback for receiving the local path
    def end_path_callback(self, msg):
        self.end_path = msg
        rospy.loginfo("Received /local_path with %d poses.", len(msg.poses))
        if self.fsm_state == FSMState.IDLE:
            rospy.loginfo("Switching to ROTATE_TO_TRAJECTORY state based on received path.")
            self.fsm_state = FSMState.ROTATE_TO_TRAJECTORY

    # Extract a local reference path segment (N+1 poses)
    def get_local_ref_path(self):
        if self.end_path is None or self.current_pose is None:
            return None

        poses = self.end_path.poses
        min_dist = float("inf")
        nearest_idx = 0
        for i, pose_st in enumerate(poses):
            p = pose_to_np(pose_st.pose)
            dist = np.linalg.norm(p[:2] - self.current_pose[:2])
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        N = 20  # Number of path segments to use
        local_poses = poses[nearest_idx:nearest_idx + N + 1]
        if len(local_poses) < N + 1:
            while len(local_poses) < N + 1:
                local_poses.append(local_poses[-1])
        ref_path = [pose_to_np(ps.pose) for ps in local_poses]
        return np.array(ref_path)

    # Compute the control input using the LQR law: u = -K * e, where
    # e = [e_x, e_theta] and u = [v, omega].
    def compute_lqr_cmd_vel(self, ref_path):
        if ref_path is None or self.current_pose is None:
            return None

        # Use the first point of the local path as the target state.
        # Calculate the desired heading from the first two points.
        ref_state = ref_path[0].copy()
        if ref_path.shape[0] > 1:
            desired_yaw = math.atan2(ref_path[1, 1] - ref_path[0, 1],
                                     ref_path[1, 0] - ref_path[0, 0])
        else:
            desired_yaw = ref_state[2]
        ref_state[2] = desired_yaw

        # Current state
        x, y, theta = self.current_pose
        dx = ref_state[0] - x
        dy = ref_state[1] - y
        # Project the positional error onto the robot's forward direction to obtain e_x
        e_x = math.cos(theta) * dx + math.sin(theta) * dy
        # Compute the heading error, normalized to [-pi, pi]
        e_theta = desired_yaw - theta
        e_theta = math.atan2(math.sin(e_theta), math.cos(e_theta))
        e = np.array([e_x, e_theta])

        # LQR control law: u = -K * e, where u = [delta_v, delta_omega]
        u = -self.lqr_K.dot(e)
        v = u[0]
        omega = u[1]

        # Saturate the commands
        v = max(min(v, self.max_lin_vel), -self.max_lin_vel)
        omega = max(min(omega, self.max_ang_vel), -self.max_ang_vel)

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = omega
        return cmd

    # Main loop: execute the FSM logic to publish control commands
    def run(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            if self.current_pose is None:
                rospy.logwarn("Current pose is not available.")
                rate.sleep()
                continue

            rospy.loginfo("Current FSM state: %s", self.fsm_state.name)

            if self.fsm_state == FSMState.ROTATE_TO_TRAJECTORY:
                # Align with the starting direction of the trajectory
                ref_path = self.get_local_ref_path()
                if ref_path is None:
                    rospy.logwarn("Reference path is not available in ROTATE_TO_TRAJECTORY state.")
                    rate.sleep()
                    continue
                if ref_path.shape[0] > 1:
                    desired_yaw = math.atan2(ref_path[1, 1] - ref_path[0, 1],
                                             ref_path[1, 0] - ref_path[0, 0])
                else:
                    desired_yaw = ref_path[0, 2]
                yaw_error = desired_yaw - self.current_pose[2]
                yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
                rospy.loginfo("Desired heading: %.2f, Current heading: %.2f, Heading error: %.2f",
                              desired_yaw, self.current_pose[2], yaw_error)
                if abs(yaw_error) < self.rot_tol:
                    rospy.loginfo("Alignment completed. Switching to LQR_TRACKING state.")
                    self.fsm_state = FSMState.LQR_TRACKING
                else:
                    twist = Twist()
                    twist.linear.x = 0
                    twist.angular.z = self.yaw_pid.compute(yaw_error, 1.0 / self.rate_hz)
                    rospy.loginfo("Publishing rotation command: linear.x=%.2f, angular.z=%.2f",
                                  twist.linear.x, twist.angular.z)
                    self.cmd_vel_pub.publish(twist)
                    rate.sleep()
                    continue

            elif self.fsm_state == FSMState.LQR_TRACKING:
                # Perform LQR tracking
                ref_path = self.get_local_ref_path()
                if ref_path is None:
                    rospy.logwarn("Reference path is not available in LQR_TRACKING state.")
                    rate.sleep()
                    continue
                cmd = self.compute_lqr_cmd_vel(ref_path)
                if cmd is not None:
                    rospy.loginfo("Publishing LQR command: linear.x=%.2f, angular.z=%.2f",
                                  cmd.linear.x, cmd.angular.z)
                    self.cmd_vel_pub.publish(cmd)
                else:
                    rospy.logwarn("Computed command is None.")

            elif self.fsm_state == FSMState.ROTATE_TO_DEFAULT:
                # Rotate to the default final orientation
                default_yaw = rospy.get_param("~default_final_yaw", 0.0)
                yaw_error = default_yaw - self.current_pose[2]
                yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
                rospy.loginfo("Final rotation: default_yaw=%.2f, current_heading=%.2f, heading error=%.2f",
                              default_yaw, self.current_pose[2], yaw_error)
                if abs(yaw_error) < self.rot_tol:
                    rospy.loginfo("Final rotation completed. Vehicle aligned.")
                    self.cmd_vel_pub.publish(Twist())
                    self.fsm_state = FSMState.IDLE
                else:
                    twist = Twist()
                    twist.linear.x = 0
                    twist.angular.z = self.yaw_pid.compute(yaw_error, 1.0 / self.rate_hz)
                    rospy.loginfo("Publishing final rotation command: linear.x=%.2f, angular.z=%.2f",
                                  twist.linear.x, twist.angular.z)
                    self.cmd_vel_pub.publish(twist)

            elif self.fsm_state == FSMState.IDLE:
                # In IDLE state, stop the vehicle
                self.cmd_vel_pub.publish(Twist())

            rate.sleep()

if __name__ == "__main__":
    try:
        tracker = LQRPathTracker()
        tracker.run()
    except rospy.ROSInterruptException:
        pass
