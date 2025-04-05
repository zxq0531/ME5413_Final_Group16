#!/usr/bin/env python3
import rospy
import math
import copy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

class LocalPlanner:
    def __init__(self):
        rospy.init_node('local_planner', anonymous=True)
        # Parameters: obstacle detection threshold, lateral offset for path modification, robot speed, horizon distance for local modification
        self.obstacle_distance_threshold = rospy.get_param("~obstacle_distance_threshold", 1.0)
        self.lateral_offset = rospy.get_param("~lateral_offset", 0.3)
        self.robot_speed = rospy.get_param("~robot_speed", 0.2)
        self.horizon_distance = rospy.get_param("~horizon_distance", 3.0)  # Only modify path points within this distance
        
        # Subscribers for global path, robot pose, and laser scan data
        rospy.Subscriber('/global_path', Path, self.global_path_callback)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)
        rospy.Subscriber('/front/scan', LaserScan, self.scan_callback)
        
        # Publisher for local path with timestamps
        self.local_path_pub = rospy.Publisher('/local_path', Path, queue_size=10)
        
        # Data storage variables
        self.global_path = None
        self.current_pose = None
        self.obstacle_detected = False
        self.obstacle_angle = None  # Angle of detected obstacle relative to the robot
        
        rospy.loginfo("Local Planner Node Initialized.")

    def global_path_callback(self, msg):
        self.global_path = msg
        rospy.loginfo("Received global path with %d poses", len(msg.poses))
    
    def pose_callback(self, msg):
        self.current_pose = msg.pose.pose

    def scan_callback(self, msg):
        # Check for obstacles within ±10 degrees in front of the robot
        angle_range = 10.0 * math.pi / 180.0
        central_ranges = []
        angles = []
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        
        for i, r in enumerate(msg.ranges):
            angle = angle_min + i * angle_increment
            if abs(angle) < angle_range:
                central_ranges.append(r)
                angles.append(angle)
                
        if central_ranges:
            min_range = min(central_ranges)
            min_index = central_ranges.index(min_range)
            self.obstacle_detected = (min_range < self.obstacle_distance_threshold)
            self.obstacle_angle = angles[min_index]
        else:
            self.obstacle_detected = False
            self.obstacle_angle = None

    def modify_path_for_obstacle(self, original_path):
        """
        When an obstacle is detected, modify the local segment of the path by applying a lateral offset 
        to the path points within a specified horizon distance. Beyond the horizon, the path remains unchanged.
        """
        # Deep copy the original path to avoid modifying global data
        new_path = copy.deepcopy(original_path)
        if self.current_pose is None:
            return new_path

        # Convert current orientation to Euler angles
        orientation_q = self.current_pose.orientation
        (_, _, yaw) = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
        
        # Only modify the local segment if an obstacle is detected
        if self.obstacle_detected and self.obstacle_angle is not None:
            # Compute the lateral offset vector (perpendicular to current heading)
            offset = self.lateral_offset
            offset_x = -offset * math.sin(yaw)
            offset_y = offset * math.cos(yaw)
            rospy.loginfo("Obstacle detected! Modifying local segment with offset: (%.2f, %.2f)", offset_x, offset_y)
            
            new_poses = []
            # Get robot's current position
            robot_x = self.current_pose.position.x
            robot_y = self.current_pose.position.y
            
            for pose_stamped in new_path.poses:
                # Calculate distance from robot to this path point
                dx = pose_stamped.pose.position.x - robot_x
                dy = pose_stamped.pose.position.y - robot_y
                distance = math.hypot(dx, dy)
                
                # Only modify points within the horizon distance
                if distance < self.horizon_distance:
                    # Check if the point is in front of the robot (within ±90° in robot frame)
                    point_angle = math.atan2(dy, dx) - yaw
                    if abs(point_angle) < math.pi/2:
                        pose_stamped.pose.position.x += offset_x
                        pose_stamped.pose.position.y += offset_y
                new_poses.append(pose_stamped)
            new_path.poses = new_poses
            
        return new_path

    def generate_timestamped_path(self, path_msg):
        """
        Generate a path with timestamps based on current time and robot speed.
        """
        stamped_path = Path()
        stamped_path.header = path_msg.header  # Preserve header info, e.g., frame_id
        stamped_path.header.stamp = rospy.Time.now()  # Update timestamp to current time
        
        new_poses = []
        total_distance = 0.0
        prev_x, prev_y = None, None
        
        for pose_stamped in path_msg.poses:
            if prev_x is not None:
                dx = pose_stamped.pose.position.x - prev_x
                dy = pose_stamped.pose.position.y - prev_y
                total_distance += math.hypot(dx, dy)
            new_pose = PoseStamped()
            new_pose.header = stamped_path.header
            new_pose.pose = pose_stamped.pose
            time_offset = rospy.Duration(total_distance / self.robot_speed)
            new_pose.header.stamp = rospy.Time.now() + time_offset
            new_poses.append(new_pose)
            prev_x = pose_stamped.pose.position.x
            prev_y = pose_stamped.pose.position.y
            
        stamped_path.poses = new_poses
        return stamped_path

    def run(self):
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            if self.global_path is not None and self.current_pose is not None:
                # If an obstacle is detected, modify the local segment of the global path;
                # otherwise, use the original global path.
                if self.obstacle_detected:
                    modified_path = self.modify_path_for_obstacle(self.global_path)
                else:
                    modified_path = self.global_path
                # Generate a timestamped local path for downstream control
                timestamped_path = self.generate_timestamped_path(modified_path)
                self.local_path_pub.publish(timestamped_path)
            rate.sleep()

if __name__ == '__main__':
    try:
        planner = LocalPlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass
