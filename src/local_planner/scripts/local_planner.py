#!/usr/bin/env python3
import rospy
import math
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray

class LocalPlannerDWA:
    def __init__(self):
        rospy.init_node('local_planner_dwa', anonymous=True)
        
        # DWA Basic Parameters
        self.max_speed = rospy.get_param("~max_speed", 0.5)  # Maximum line speed
        self.max_angular_speed = rospy.get_param("~max_angular_speed", 1.0)  # Maximum angular velocity
        self.acceleration = rospy.get_param("~acceleration", 0.2)  # Maximum acceleration
        self.angular_acceleration = rospy.get_param("~angular_acceleration", 0.5)  # Maximum angular acceleration
        self.time_horizon = rospy.get_param("~time_horizon", 2.0)  # Projection time
        self.dt = rospy.get_param("~dt", 0.1)  # time step
        self.obstacle_distance_threshold = rospy.get_param("~obstacle_distance_threshold", 0.5)  # Obstacle Distance Threshold
        self.goal_tolerance = rospy.get_param("~goal_tolerance", 0.2)  # Tolerance distance to target point
        # Scoring weight parameters
        self.goal_weight = rospy.get_param("~goal_weight", 0.6)  # Target scoring weights
        self.obstacle_weight = rospy.get_param("~obstacle_weight", 2.0)  # Obstacle scoring weights
        self.speed_weight = rospy.get_param("~speed_weight", 0.2)  # Speed score weighting
        self.heading_weight = rospy.get_param("~heading_weight", 0.8)  # Heading score weights
        self.clearance_weight = rospy.get_param("~clearance_weight", 1.5)  # Gap scoring weights
        
        # Speed Sampling Parameters
        self.v_samples = rospy.get_param("~v_samples", 11)  # Number of Linear Velocity Samples
        self.w_samples = rospy.get_param("~w_samples", 15)  # Number of angular velocity samples
        self.min_speed = rospy.get_param("~min_speed", 0.0)  # Minimum line speed
        
        # Path tracing parameters
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 1.0)  # forward looking distance
        
        # Barrier Avoidance Parameters
        self.lethal_cost_radius = rospy.get_param("~lethal_cost_radius", 0.3)  # Fatal cost radius

        # Subscription to global path, robot position and LiDAR data
        rospy.Subscriber('/global_path', Path, self.global_path_callback)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)
        rospy.Subscriber('/front/scan', LaserScan, self.scan_callback)

        # Issuing control commands and localized paths
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.local_path_pub = rospy.Publisher('/local_path', Path, queue_size=10)
        self.trajectory_markers_pub = rospy.Publisher('/trajectory_markers', MarkerArray, queue_size=10)
        self.obstacle_markers_pub = rospy.Publisher('/obstacle_markers', MarkerArray, queue_size=10)

        # data storage
        self.global_path = None
        self.current_pose = None
        self.scan_data = None
        self.current_v = 0.0
        self.current_w = 0.0
        self.velocity_history = []
        self.obstacle_points = []
        self.target_idx = 0

        rospy.loginfo("DWA Local Planner Node Initialized.")

    def global_path_callback(self, msg):
        self.global_path = msg
        self.target_idx = 0
        rospy.loginfo("Received global path with %d poses", len(msg.poses))
    
    def pose_callback(self, msg):
        self.current_pose = msg.pose.pose
        # Updating the target point index
        if self.global_path is not None and len(self.global_path.poses) > 0:
            self.target_idx = self.find_target_idx()

    def scan_callback(self, msg):
        self.scan_data = msg
        self.obstacle_points = self.get_obstacle_points(msg)
        self.publish_obstacle_markers(self.obstacle_points)

    def find_target_idx(self):
        """
        Find a suitable destination index on the global path
        """
        if self.current_pose is None or self.global_path is None:
            return 0
            
        robot_x = self.current_pose.position.x
        robot_y = self.current_pose.position.y
        
        # Start searching from the current index
        min_dist = float('inf')
        min_idx = self.target_idx
        
        # First find the nearest point
        for i in range(len(self.global_path.poses)):
            pose = self.global_path.poses[i].pose
            dist = math.hypot(pose.position.x - robot_x, pose.position.y - robot_y)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        
        # Then move forward to find the right front viewpoint
        lookahead_idx = min_idx
        lookahead_dist = 0.0
        
        for i in range(min_idx, len(self.global_path.poses)):
            if i == min_idx:
                continue
                
            pose = self.global_path.poses[i].pose
            prev_pose = self.global_path.poses[i-1].pose
            
            # Calculate the distance of the current segment
            segment_dist = math.hypot(
                pose.position.x - prev_pose.position.x,
                pose.position.y - prev_pose.position.y
            )
            
            lookahead_dist += segment_dist
            
            if lookahead_dist >= self.lookahead_distance:
                lookahead_idx = i
                break
        
        return lookahead_idx

    def get_obstacle_points(self, scan_data):
        """
        Converting laser scan data to obstacle points in a global coordinate system
        """
        if self.current_pose is None or scan_data is None:
            return []
        
        obstacle_points = []
        robot_x = self.current_pose.position.x
        robot_y = self.current_pose.position.y
        orientation_q = self.current_pose.orientation
        (_, _, yaw) = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
        
        angle = scan_data.angle_min
        for r in scan_data.ranges:
            if r < scan_data.range_max and r > scan_data.range_min:
                # Points in the LiDAR coordinate system
                local_x = r * math.cos(angle)
                local_y = r * math.sin(angle)
                
                # Convert to global coordinate system
                global_x = robot_x + local_x * math.cos(yaw) - local_y * math.sin(yaw)
                global_y = robot_y + local_x * math.sin(yaw) + local_y * math.cos(yaw)
                
                obstacle_points.append((global_x, global_y))
            
            angle += scan_data.angle_increment
        
        return obstacle_points
    
    def publish_obstacle_markers(self, obstacle_points):
        """
        Publishing visual markers for obstacle points
        """
        marker_array = MarkerArray()
        
        for i, (x, y) in enumerate(obstacle_points):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "obstacles"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.1
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 0.8
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)
        
        self.obstacle_markers_pub.publish(marker_array)

    def velocity_range(self, current_v=0.0):
        """
        Generate linear velocity sampling range considering current velocity and acceleration constraints
        """
        v_min = max(self.min_speed, current_v - self.acceleration * self.dt)
        v_max = min(self.max_speed, current_v + self.acceleration * self.dt)
        
        # Generate a more reasonable number of sampling points
        if self.v_samples > 1:
            return np.linspace(v_min, v_max, self.v_samples)
        else:
            return [v_min]

    def angular_velocity_range(self, current_w=0.0):
        """
        Generate angular velocity sampling range considering current angular velocity and angular acceleration constraints
        """
        w_min = max(-self.max_angular_speed, current_w - self.angular_acceleration * self.dt)
        w_max = min(self.max_angular_speed, current_w + self.angular_acceleration * self.dt)
        
        # Generate a more reasonable number of sampling points
        if self.w_samples > 1:
            return np.linspace(w_min, w_max, self.w_samples)
        else:
            return [w_min]

    def predict_trajectory(self, v, w, x, y, yaw, time_horizon, dt):
        """
        Predicts trajectories and simulates future positions based on current velocity and angular velocity.
        """
        trajectory = []
        for t in range(0, int(time_horizon / dt)):
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
            yaw += w * dt
            trajectory.append((x, y, yaw))
        return trajectory

    def calc_goal_score(self, trajectory, goal_x, goal_y):
        """
        Calculate the distance score between the end point of the trajectory and the target point
        """
        if not trajectory:
            return -float('inf')
            
        final_x, final_y, _ = trajectory[-1]
        goal_distance = math.hypot(final_x - goal_x, final_y - goal_y)
        
        # The smaller the distance, the higher the score
        return -goal_distance

    def calc_obstacle_score(self, trajectory, obstacle_points):
        """
        Calculate the distance score of the trajectory from the obstacle
        """
        if not trajectory or not obstacle_points:
            return 0.0
            
        min_dist = float('inf')
        
        for (x, y, _) in trajectory:
            for (obs_x, obs_y) in obstacle_points:
                dist = math.hypot(x - obs_x, y - obs_y)
                if dist < min_dist:
                    min_dist = dist
        
        # If the minimum distance is less than the threshold, give a penalty
        if min_dist < self.obstacle_distance_threshold:
            return -1.0 / (min_dist + 0.001)  # Avoiding division by zero errors
        else:
            return 0.0

    def calc_speed_score(self, v):
        """
        Calculate speed scores to encourage higher speeds
        """
        # The closer the speed is to the maximum speed, the higher the score
        return v / self.max_speed

    def calc_heading_score(self, trajectory, goal_x, goal_y):
        """
        Calculate heading scores to encourage heading toward the target point
        """
        if not trajectory:
            return -float('inf')
            
        final_x, final_y, final_yaw = trajectory[-1]
        
        # Calculate the angle from the end of the trajectory to the target
        goal_angle = math.atan2(goal_y - final_y, goal_x - final_x)
        
        # Calculate the angular difference
        angle_diff = abs(self.normalize_angle(goal_angle - final_yaw))
        
        # The smaller the angle difference, the higher the score
        return 1.0 - angle_diff / math.pi

    def calc_clearance_score(self, trajectory, obstacle_points):
        """
        Calculate the average gap score between the trajectory and the obstacle
        """
        if not trajectory or not obstacle_points:
            return 0.0
            
        total_clearance = 0.0
        min_clearance = float('inf')
        
        for (x, y, _) in trajectory:
            for (obs_x, obs_y) in obstacle_points:
                dist = math.hypot(x - obs_x, y - obs_y)
                if dist < min_clearance:
                    min_clearance = dist
            
            # Cumulative Gap Fraction
            if min_clearance < self.obstacle_distance_threshold:
                total_clearance += min_clearance / self.obstacle_distance_threshold
            else:
                total_clearance += 1.0
        
        # Returns the average gap score
        return total_clearance / len(trajectory) if trajectory else 0.0

    def normalize_angle(self, angle):
        """
        Normalize the angle to the range [-pi, pi]
        """
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def dwa_planning(self):
        """
        DWA algorithm implementation for calculating optimal velocity combinations
          (linear and angular velocities) as well as local trajectories.
        """
        if self.global_path is None or self.current_pose is None or self.scan_data is None:
            return None, None, []

        # Get the current position and orientation of the robot
        robot_x = self.current_pose.position.x
        robot_y = self.current_pose.position.y
        orientation_q = self.current_pose.orientation
        (_, _, yaw) = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])

        # Acquisition of target points
        if self.target_idx >= len(self.global_path.poses):
            rospy.loginfo("Reached end of path")
            return 0.0, 0.0, []
            
        goal_pose = self.global_path.poses[self.target_idx].pose
        goal_x = goal_pose.position.x
        goal_y = goal_pose.position.y

        # Check if the target point has been reached
        dist_to_goal = math.hypot(goal_x - robot_x, goal_y - robot_y)
        if dist_to_goal < self.goal_tolerance:
            self.target_idx += 1
            if self.target_idx >= len(self.global_path.poses):
                rospy.loginfo("Reached final goal")
                return 0.0, 0.0, []
            goal_pose = self.global_path.poses[self.target_idx].pose
            goal_x = goal_pose.position.x
            goal_y = goal_pose.position.y

        # Dynamic window calculation
        v_samples = self.velocity_range(self.current_v)
        w_samples = self.angular_velocity_range(self.current_w)

        # Initialize the optimal speed and score
        best_score = float('-inf')
        best_v = 0.0
        best_w = 0.0
        best_trajectory = []
        all_trajectories = []

        # Evaluate all possible speed combinations
        for v in v_samples:
            for w in w_samples:
                # Predicted trajectory
                trajectory = self.predict_trajectory(v, w, robot_x, robot_y, yaw, self.time_horizon, self.dt)
                all_trajectories.append((trajectory, v, w))
                
                # Calculation of individual scores
                goal_score = self.calc_goal_score(trajectory, goal_x, goal_y)
                obstacle_score = self.calc_obstacle_score(trajectory, self.obstacle_points)
                speed_score = self.calc_speed_score(v)
                heading_score = self.calc_heading_score(trajectory, goal_x, goal_y)
                clearance_score = self.calc_clearance_score(trajectory, self.obstacle_points)
                
                # overall rating
                score = (
                    goal_score * self.goal_weight +
                    obstacle_score * self.obstacle_weight +
                    speed_score * self.speed_weight +
                    heading_score * self.heading_weight +
                    clearance_score * self.clearance_weight
                )
                
                # Updating optimal speed combinations and trajectories
                if score > best_score:
                    best_score = score
                    best_v = v
                    best_w = w
                    best_trajectory = trajectory

        # Publish all trajectories for visualization
        self.publish_trajectory_markers(all_trajectories, best_trajectory)

        # Save current speed for next time
        self.current_v = best_v
        self.current_w = best_w

        return best_v, best_w, best_trajectory

    def publish_trajectory_markers(self, all_trajectories, best_trajectory):
        """
        Publishing trajectory visualization markers
        """
        marker_array = MarkerArray()
        
        # Add all candidate tracks
        for i, (trajectory, v, w) in enumerate(all_trajectories):
            if not trajectory:
                continue
                
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "trajectories"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.02 
            marker.color.a = 0.3 
            marker.color.r = 0.0
            marker.color.g = 0.7
            marker.color.b = 0.7
            
            for (x, y, _) in trajectory:
                p = geometry_msgs.msg.Point()
                p.x = x
                p.y = y
                p.z = 0.1
                marker.points.append(p)
                
            marker_array.markers.append(marker)
        
        # Adding the best trajectory
        if best_trajectory:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "trajectories"
            marker.id = len(all_trajectories)
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.05
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            
            for (x, y, _) in best_trajectory:
                p = geometry_msgs.msg.Point()
                p.x = x
                p.y = y
                p.z = 0.1
                marker.points.append(p)
                
            marker_array.markers.append(marker)
        
        self.trajectory_markers_pub.publish(marker_array)

    def publish_local_path(self, trajectory):
        """
        Post local traces to the /local_path topic
        """
        local_path = Path()
        local_path.header.frame_id = "map"
        local_path.header.stamp = rospy.Time.now()

        for (x, y, yaw) in trajectory:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            local_path.poses.append(pose)

        self.local_path_pub.publish(local_path)

    def avoid_oscillation(self, v, w):
        """
        Avoid oscillations by recording historical tracks
        """
        # Maintain a speed history of the last N
        if len(self.velocity_history) > 10:
            self.velocity_history.pop(0)
        
        # Add current speed to history
        self.velocity_history.append((v, w))
        
        # Check for the presence of an oscillation mode
        if len(self.velocity_history) >= 6:
            oscillation_detected = True
            for i in range(3):
                if (self.velocity_history[-1-i][1] * self.velocity_history[-4-i][1] > 0):
                    oscillation_detected = False
                    break
            
            if oscillation_detected:
                return v, w * 0.5
        
        return v, w

    def run(self):
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            best_v, best_w, best_trajectory = self.dwa_planning()
            if best_trajectory:
                best_v, best_w = self.avoid_oscillation(best_v, best_w)
                final_v = 1.5 * best_v
                final_w = 1.5 * best_w

                cmd = Twist()
                cmd.linear.x = final_v
                cmd.angular.z = final_w
                self.cmd_vel_pub.publish(cmd)

                self.publish_local_path(best_trajectory)
            rate.sleep()

if __name__ == '__main__':
    try:
        import geometry_msgs.msg
        planner = LocalPlannerDWA()
        planner.run()
    except rospy.ROSInterruptException:
        pass
