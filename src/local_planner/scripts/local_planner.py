#!/usr/bin/env python3
import rospy
import math
import copy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

class LocalPlannerDWA:
    def __init__(self):
        rospy.init_node('local_planner_dwa', anonymous=True)
        
        # DWA参数
        self.max_speed = rospy.get_param("~max_speed", 0.5)  # 最大线速度
        self.max_angular_speed = rospy.get_param("~max_angular_speed", 1.0)  # 最大角速度
        self.acceleration = rospy.get_param("~acceleration", 0.2)  # 最大加速度
        self.angular_acceleration = rospy.get_param("~angular_acceleration", 0.5)  # 最大角加速度
        self.time_horizon = rospy.get_param("~time_horizon", 3.0)  # 预测时间
        self.dt = rospy.get_param("~dt", 0.1)  # 时间步长
        self.obstacle_distance_threshold = rospy.get_param("~obstacle_distance_threshold", 1.0)  # 障碍物距离阈值
        self.goal_tolerance = rospy.get_param("~goal_tolerance", 0.2)  # 到达目标点的容忍距离

        # 订阅全局路径、机器人位姿和激光雷达数据
        rospy.Subscriber('/global_path', Path, self.global_path_callback)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)
        rospy.Subscriber('/front/scan', LaserScan, self.scan_callback)

        # 发布局部路径
        self.local_path_pub = rospy.Publisher('/local_path', Path, queue_size=10)

        # 数据存储
        self.global_path = None
        self.current_pose = None
        self.scan_data = None

        rospy.loginfo("DWA Local Planner Node Initialized.")

    def global_path_callback(self, msg):
        self.global_path = msg
        rospy.loginfo("Received global path with %d poses", len(msg.poses))
    
    def pose_callback(self, msg):
        self.current_pose = msg.pose.pose

    def scan_callback(self, msg):
        self.scan_data = msg

    def dwa_planning(self):
        """
        DWA算法实现，用于计算最优速度组合（线速度和角速度）。
        """
        if self.global_path is None or self.current_pose is None or self.scan_data is None:
            return None

        # 获取机器人当前位置和朝向
        robot_x = self.current_pose.position.x
        robot_y = self.current_pose.position.y
        orientation_q = self.current_pose.orientation
        (_, _, yaw) = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])

        # 获取目标点（全局路径的下一个点）
        goal_pose = self.global_path.poses[0].pose
        goal_x = goal_pose.position.x
        goal_y = goal_pose.position.y

        # 初始化最优速度和评分
        best_score = float('-inf')
        best_v = 0.0
        best_w = 0.0

        # 动态窗口计算
        for v in self.velocity_range(self.max_speed, self.acceleration, self.dt):
            for w in self.angular_velocity_range(self.max_angular_speed, self.angular_acceleration, self.dt):
                # 预测轨迹
                trajectory = self.predict_trajectory(v, w, robot_x, robot_y, yaw, self.time_horizon, self.dt)
                
                # 评分函数
                score = self.evaluate_trajectory(trajectory, goal_x, goal_y, self.scan_data)
                
                # 更新最优速度组合
                if score > best_score:
                    best_score = score
                    best_v = v
                    best_w = w

        # 根据最优速度生成局部路径
        local_path = self.generate_local_path(best_v, best_w, robot_x, robot_y, yaw)

        return local_path

    def velocity_range(self, max_speed, acceleration, dt):
        return np.linspace(0, max_speed, int(max_speed / acceleration / dt) + 1)

    def angular_velocity_range(self, max_angular_speed, angular_acceleration, dt):
        return np.linspace(-max_angular_speed, max_angular_speed, int(max_angular_speed / angular_acceleration / dt) + 1)

    def predict_trajectory(self, v, w, x, y, yaw, time_horizon, dt):
        trajectory = []
        for t in range(0, int(time_horizon / dt)):
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
            yaw += w * dt
            trajectory.append((x, y, yaw))
        return trajectory

    def evaluate_trajectory(self, trajectory, goal_x, goal_y, scan_data):
        final_x, final_y, _ = trajectory[-1]
        goal_distance = math.hypot(final_x - goal_x, final_y - goal_y)
        goal_score = -goal_distance

        obstacle_score = 0.0
        for (x, y, _) in trajectory:
            for i, r in enumerate(scan_data.ranges):
                if r < self.obstacle_distance_threshold:
                    obstacle_score -= 1.0 / r

        return goal_score + obstacle_score

    def generate_local_path(self, v, w, x, y, yaw):
        local_path = Path()
        local_path.header.stamp = rospy.Time.now()
        local_path.header.frame_id = "map"

        # 使用全局路径的时间戳为局部路径赋值
        for i, pose_stamped in enumerate(self.global_path.poses):
            pose = PoseStamped()
            pose.header = local_path.header
            pose.pose.position.x = x + v * math.cos(yaw) * i * self.dt
            pose.pose.position.y = y + v * math.sin(yaw) * i * self.dt
            pose.pose.orientation.w = 1.0
            pose.header.stamp = pose_stamped.header.stamp  # 对应全局路径的时间戳
            local_path.poses.append(pose)

        return local_path

    def run(self):
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            local_path = self.dwa_planning()
            if local_path is not None:
                self.local_path_pub.publish(local_path)
            rate.sleep()

if __name__ == '__main__':
    try:
        planner = LocalPlannerDWA()
        planner.run()
    except rospy.ROSInterruptException:
        pass
