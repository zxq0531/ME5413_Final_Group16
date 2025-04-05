#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import rospy
import rospkg
import cv2
import numpy as np
import yaml
import heapq
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped

# --------------------- A* GLOBAL PATH PLANNING ---------------------
class Node:
    def __init__(self, x, y, cost, priority, parent=None):
        self.x = x                # x-coordinate in the grid
        self.y = y                # y-coordinate in the grid
        self.cost = cost          # Cost from the start node (g value)
        self.priority = priority  # f value = g + h (cost + heuristic)
        self.parent = parent      # Parent node for path backtracking

    def __lt__(self, other):
        return self.priority < other.priority

def heuristic(x, y, goal):
    return np.sqrt((x - goal[0])**2 + (y - goal[1])**2)

def a_star(occupancy_grid, start, goal):
    """
    Performs A* search on the occupancy grid to generate a corridor path.
    """
    height, width = occupancy_grid.shape
    open_list = []
    closed_set = set()

    start_node = Node(start[0], start[1], 0, heuristic(start[0], start[1], goal))
    heapq.heappush(open_list, start_node)

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_list:
        current = heapq.heappop(open_list)
        if (current.x, current.y) in closed_set:
            continue

        if (current.x, current.y) == (goal[0], goal[1]):
            path = []
            while current is not None:
                path.append((current.x, current.y))
                current = current.parent
            path.reverse()
            return path

        closed_set.add((current.x, current.y))
        for dx, dy in neighbors:
            nx, ny = current.x + dx, current.y + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if occupancy_grid[ny, nx] == 0:
                continue
            if (nx, ny) in closed_set:
                continue
            move_cost = np.sqrt(2) if dx != 0 and dy != 0 else 1
            new_cost = current.cost + move_cost
            priority = new_cost + heuristic(nx, ny, goal)
            new_node = Node(nx, ny, new_cost, priority, current)
            heapq.heappush(open_list, new_node)
    return None

# --------------------- BOUSTROPHEDON COVERAGE PATH WITH CHAIKIN SMOOTHING ---------------------
def boustrophedon_path(occupancy_grid, start, goal, row_step=20):
    """
    Generates a boustrophedon (ox-plowing) path between the start and goal points.
    Handles descending order if the start's y-coordinate is greater than the goal's.
    """
    x_min = min(start[0], goal[0])
    x_max = max(start[0], goal[0])
    y_min = min(start[1], goal[1])
    y_max = max(start[1], goal[1])

    path = [start]
    if start[1] >= goal[1]:
        y_range = range(start[1], y_min, -row_step)
    else:
        y_range = range(start[1], y_max, row_step)

    direction = 1  # 1: moving rightward, -1: moving leftward
    for y in y_range:
        row = occupancy_grid[y, x_min:x_max+1]
        free_indices = np.where(row == 1)[0]
        if len(free_indices) == 0:
            continue
        if direction == 1:
            x_waypoint = x_min + free_indices[-1]
        else:
            x_waypoint = x_min + free_indices[0]
        path.append((x_waypoint, y))
        direction *= -1
    path.append(goal)
    return path

def chaikin_smoothing(path, iterations=3):
    """
    Smooths the given path using Chaikin's corner-cutting algorithm.
    """
    for _ in range(iterations):
        new_path = [path[0]]
        for i in range(len(path) - 1):
            p0 = np.array(path[i], dtype=float)
            p1 = np.array(path[i+1], dtype=float)
            Q = 0.75 * p0 + 0.25 * p1
            R = 0.25 * p0 + 0.75 * p1
            new_path.append(tuple(Q))
            new_path.append(tuple(R))
        new_path.append(path[-1])
        path = new_path
    return [(int(round(pt[0])), int(round(pt[1]))) for pt in path]

# --------------------- MAP LOADING FUNCTION ---------------------
def load_map(yaml_file):
    """
    Loads the map image and its corresponding YAML metadata.
    Derives the image file path relative to the YAML file.
    """
    with open(yaml_file, 'r') as f:
        map_yaml = yaml.safe_load(f)

    # The YAML file should contain an "image" field, e.g., "my_map.pgm"
    yaml_dir = os.path.dirname(yaml_file)
    image_path = os.path.join(yaml_dir, map_yaml["image"])

    map_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if map_img is None:
        raise IOError("Failed to load map image file: " + image_path)

    return map_img, map_yaml

# --------------------- OBSTACLE INFLATION ---------------------
def inflate_obstacles(occupancy_grid, kernel_size=(10,10)):
    """
    Inflates obstacles in the occupancy grid to account for the robot's width.
    Uses a kernel of the specified size to dilate obstacles.
    """
    # Invert grid: obstacles become 1, free space becomes 0
    inverted = np.where(occupancy_grid == 0, 1, 0).astype(np.uint8) * 255
    kernel = np.ones(kernel_size, np.uint8)
    dilated = cv2.dilate(inverted, kernel, iterations=1)
    # Invert back: if a dilated pixel is > 0, mark as obstacle (0); otherwise free (1)
    inflated_grid = np.where(dilated > 0, 0, 1).astype(np.uint8)
    return inflated_grid

# --------------------- VISUALIZATION FUNCTION ---------------------
def visualize_path(orig_img, complete_path):
    """
    Visualizes the complete trajectory on the map.
    """
    vis_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
    for i, point in enumerate(complete_path):
        cv2.circle(vis_img, point, 1, (0, 0, 255), -1)
        if i > 0:
            cv2.line(vis_img, complete_path[i-1], point, (0, 0, 255), 1)
    cv2.imshow("Complete Trajectory", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --------------------- Global Variables and Callback Function ---------------------
robot_origin = None  # Stores the robot's starting point (grid coordinates)
map_info_global = None  # Stores the loaded map metadata

def amcl_pose_callback(msg):
    global robot_origin, map_info_global
    # Retrieve the robot's position from the /amcl_pose topic
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    # Convert the position to grid coordinates using the map's origin and resolution from the YAML file
    origin = map_info_global['origin']  # [origin_x, origin_y, theta]
    resolution = map_info_global['resolution']
    grid_x = int(round((x - origin[0]) / resolution))
    grid_y = int(round((y - origin[1]) / resolution))
    robot_origin = (grid_x, grid_y)
    rospy.loginfo("Robot origin (grid): %s", str(robot_origin))

# --------------------- MAIN EXECUTION ---------------------
if __name__ == "__main__":
    rospy.init_node('global_planner_node', anonymous=True)
    path_pub = rospy.Publisher('/global_path', Path, queue_size=10)
    rate = rospy.Rate(1)  # 1 Hz

    # Use rospkg to get the path of the me5413_world package and construct the map YAML file path.
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path("me5413_world")
    yaml_file = os.path.join(pkg_path, "maps", "my_map.yaml")
    rospy.loginfo("Using map YAML file: %s", yaml_file)

    # Load the map image and YAML metadata
    occupancy_img, map_info = load_map(yaml_file)
    map_info_global = map_info  # Save globally for callback usage

    ret, binary_img = cv2.threshold(occupancy_img, 250, 255, cv2.THRESH_BINARY)
    occupancy_grid = (binary_img // 255).astype(np.uint8)

    # Inflate obstacles using a 10x10 kernel
    inflated_grid = inflate_obstacles(occupancy_grid, kernel_size=(10,10))

    # Subscribe to /amcl_pose to obtain the robot's origin
    rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, amcl_pose_callback)

    # Wait until the robot's origin is received from /amcl_pose
    rospy.loginfo("Waiting for robot origin from /amcl_pose...")
    while robot_origin is None and not rospy.is_shutdown():
        rospy.sleep(0.1)

    # Use the robot's origin as the starting point for A* path planning
    corridor_start = (robot_origin[0], 776 - robot_origin[1])

    # The goal remains as before (modify as needed)
    corridor_goal  = (485, 475)

    # Execute A* path planning
    corridor_path = a_star(inflated_grid, corridor_start, corridor_goal)
    if corridor_path is None:
        rospy.logerr("A* Corridor Path not found.")
        exit(1)
    rospy.loginfo("A* Corridor Path computed.")

    # Generate Boustrophedon coverage path starting from the end of the A* path
    coverage_start = corridor_path[-1]
    # coverage_goal = (230, 700)
    coverage_goal  = (310, 100)
    
    coverage_path = boustrophedon_path(inflated_grid, coverage_start, coverage_goal, row_step=45)
    if coverage_path is None:
        rospy.logerr("Boustrophedon Coverage Path not found.")
        exit(1)
    rospy.loginfo("Boustrophedon Coverage Path computed.")

    # Smooth the coverage path using Chaikin's algorithm
    smoothed_coverage = chaikin_smoothing(coverage_path, iterations=4)

    # Concatenate the two paths (removing the duplicate junction point)
    complete_trajectory = corridor_path + smoothed_coverage[1:]
    
    # After computing the complete trajectory in pixel coordinates
    # First, flip y coordinate: new_y = 776 - old_y
    flipped_trajectory = [(pt[0], 776 - pt[1]) for pt in complete_trajectory]

    # Then, convert from pixel coordinates to world coordinates using resolution and origin
    resolution = map_info_global['resolution']  # e.g., 0.05
    origin = map_info_global['origin']          # e.g., [-6.55, -36.95, 0]
    world_trajectory = [(
        pt[0] * resolution + origin[0],
        pt[1] * resolution + origin[1]
    ) for pt in flipped_trajectory]

    rospy.loginfo("Complete trajectory generated with %d points.", len(world_trajectory))


    # Build the nav_msgs/Path message
    path_msg = Path()
    path_msg.header.stamp = rospy.Time.now()
    path_msg.header.frame_id = "map"  # Ensure this frame matches the one used in RViz

    for pt in world_trajectory:
        pose = PoseStamped()
        pose.header = path_msg.header
        pose.pose.position.x = pt[0]
        pose.pose.position.y = pt[1]
        pose.pose.position.z = 0
        pose.pose.orientation.w = 1.0  # Default orientation
        path_msg.poses.append(pose)

    # Continuously publish the path message until node shutdown
    while not rospy.is_shutdown():
        path_msg.header.stamp = rospy.Time.now()  # Update timestamp
        path_pub.publish(path_msg)
        rate.sleep()
    