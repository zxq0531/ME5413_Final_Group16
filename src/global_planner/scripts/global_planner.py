#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import math
import rospy
import rospkg
import cv2
import numpy as np
import yaml
import heapq
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from scipy.interpolate import splprep, splev

# --------------------- A* GLOBAL PATH PLANNING ---------------------
class Node:
    def __init__(self, x, y, cost, priority, parent=None):
        self.x = x                # x-coordinate in the grid
        self.y = y                # y-coordinate in the grid
        self.cost = cost          # Cost from the start node (g value)
        self.priority = priority  # f = g + h (cost + heuristic)
        self.parent = parent      # Parent node for backtracking

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

# --------------------- BOUSTROPHEDON COVERAGE PATH PLANNING ---------------------
def boustrophedon_path(occupancy_grid, start, goal, row_step=40):
    """
    Generates a boustrophedon (lawn-mowing) coverage path between the start and goal points.
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

# --------------------- SPLINE SMOOTHING FUNCTION ---------------------
def spline_smoothing(path, num_points=150, smoothing=0):
    """
    Smooths the given path using spline interpolation.
    
    Parameters:
        path: List of (x, y) tuples representing the original path.
        num_points: Number of points to sample in the smooth path.
        smoothing: Smoothing factor; s=0 forces interpolation through all points.
        
    Returns:
        smooth_path: List of (x, y) tuples (integer coordinates) representing the smooth path.
    """
    path = np.array(path)
    if len(path) < 2:
        return path.tolist()
    # Create a parametric spline representation of the path
    tck, u = splprep([path[:,0], path[:,1]], s=smoothing)
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)
    smooth_path = [(int(round(x)), int(round(y))) for x, y in zip(x_new, y_new)]
    return smooth_path

# --------------------- MAP LOADING FUNCTION ---------------------
def load_map(yaml_file):
    """
    Loads the map image and its corresponding YAML metadata.
    Derives the image file path relative to the YAML file.
    """
    with open(yaml_file, 'r') as f:
        map_yaml = yaml.safe_load(f)

    yaml_dir = os.path.dirname(yaml_file)
    image_path = os.path.join(yaml_dir, map_yaml["image"])

    map_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if map_img is None:
        raise IOError("Failed to load map image file: " + image_path)

    return map_img, map_yaml

# --------------------- OBSTACLE INFLATION FUNCTION ---------------------
def inflate_obstacles(occupancy_grid, kernel_size=(20,20)):
    """
    Inflates obstacles in the occupancy grid to account for the robot's width.
    Uses a kernel of the specified size to dilate obstacles.
    """
    inverted = np.where(occupancy_grid == 0, 1, 0).astype(np.uint8) * 255
    kernel = np.ones(kernel_size, np.uint8)
    dilated = cv2.dilate(inverted, kernel, iterations=1)
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

# --------------------- Global Variables and Callback ---------------------
robot_origin = None  # Stores the robot's starting point (grid coordinates)
map_info_global = None  # Stores the loaded map metadata

def amcl_pose_callback(msg):
    global robot_origin, map_info_global
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
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

    # Get package path and construct the map YAML file path
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path("me5413_world")
    yaml_file = os.path.join(pkg_path, "maps", "my_map.yaml")
    rospy.loginfo("Using map YAML file: %s", yaml_file)

    # Load the map image and YAML metadata
    occupancy_img, map_info = load_map(yaml_file)
    map_info_global = map_info  # Save globally for callback usage

    ret, binary_img = cv2.threshold(occupancy_img, 0, 255, cv2.THRESH_BINARY)
    occupancy_grid = (binary_img // 255).astype(np.uint8)

    # Inflate obstacles using a 20x20 kernel
    inflated_grid = inflate_obstacles(occupancy_grid, kernel_size=(20,20))

    # Subscribe to /amcl_pose to obtain the robot's origin
    rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, amcl_pose_callback)
    rospy.loginfo("Waiting for robot origin from /amcl_pose...")
    while robot_origin is None and not rospy.is_shutdown():
        rospy.sleep(0.1)

    # Use the robot's origin as the starting point for A* path planning
    # Adjust coordinate transformation as needed (e.g., flipping y axis)
    corridor_start = (robot_origin[0], 508 - robot_origin[1])
    corridor_goal  = (425, 455)

    # Execute A* path planning for the corridor
    corridor_path = a_star(inflated_grid, corridor_start, corridor_goal)
    if corridor_path is None:
        rospy.logerr("A* Corridor Path not found.")
        exit(1)
    rospy.loginfo("A* Corridor Path computed.")

    # Smooth the A* corridor path using spline interpolation
    smoothed_corridor = spline_smoothing(corridor_path, num_points=150, smoothing=0)
    rospy.loginfo("Smoothed A* Corridor Path computed.")

    # Set coverage goal to (300,95) so that the extra segment can start from (300,95)
    coverage_start = corridor_path[-1]
    coverage_goal  = (300, 95)
    coverage_path = boustrophedon_path(inflated_grid, coverage_start, coverage_goal, row_step=40)
    if coverage_path is None:
        rospy.logerr("Boustrophedon Coverage Path not found.")
        exit(1)
    rospy.loginfo("Boustrophedon Coverage Path computed.")

    # Smooth the coverage path using Chaikin's algorithm
    smoothed_coverage = chaikin_smoothing(coverage_path, iterations=4)

    # Concatenate the two paths (removing the duplicate junction point)
    complete_trajectory = smoothed_corridor + smoothed_coverage[1:]
    rospy.loginfo("Complete trajectory computed with %d points.", len(complete_trajectory))

    # --------------------- New Segment: Smooth Arc Transition and Straight Line Along River ---------------------
    # Generate an arc from (300,95) to (260,135)
    arc_center = (300, 135)  # Center chosen so that (300,95) lies on the arc
    arc_radius = 40
    num_arc_points = 50  # Number of sample points on the arc
    theta_values = np.linspace(-np.pi/2, -np.pi, num_arc_points)
    arc_points = []
    for theta in theta_values:
        x = int(round(arc_center[0] + arc_radius * math.cos(theta)))
        y = int(round(arc_center[1] + arc_radius * math.sin(theta)))
        arc_points.append((x, y))

    # Generate a straight-line segment along the river (x fixed at 260) from the arc end to (260,455)
    line_start = arc_points[-1]  # Expected to be (260,135)
    line_end = (260, 460)
    num_line_points = 50  # Number of sample points on the line
    line_points = []
    for i in range(1, num_line_points + 1):
        y = int(round(line_start[1] + (line_end[1] - line_start[1]) * i / num_line_points))
        line_points.append((260, y))

    # Combine the new segment (skip the duplicate starting point)
    new_segment = arc_points + line_points
    complete_trajectory_extended = complete_trajectory + new_segment[1:]
    rospy.loginfo("Extended complete trajectory computed with %d points.", len(complete_trajectory_extended))

    # --------------------- Convert Trajectory to World Coordinates and Publish ---------------------
    # Flip y-axis coordinates and convert pixel coordinates to world coordinates
    flipped_trajectory = [(pt[0], 508 - pt[1]) for pt in complete_trajectory_extended]
    resolution = map_info_global['resolution']  # e.g., 0.05
    origin = map_info_global['origin']          # e.g., [-6.55, -36.95, 0]
    world_trajectory = [(
        pt[0] * resolution + origin[0],
        pt[1] * resolution + origin[1]
    ) for pt in flipped_trajectory]

    # Build the nav_msgs/Path message
    path_msg = Path()
    path_msg.header.stamp = rospy.Time.now()
    path_msg.header.frame_id = "map"

    for pt in world_trajectory:
        pose = PoseStamped()
        pose.header = path_msg.header
        pose.pose.position.x = pt[0]
        pose.pose.position.y = pt[1]
        pose.pose.position.z = 0
        pose.pose.orientation.w = 1.0
        path_msg.poses.append(pose)

    # Optionally visualize the complete path in pixel space (uncomment if needed)
    # visualize_path(occupancy_img, complete_trajectory_extended)

    # Continuously publish the path message until node shutdown
    while not rospy.is_shutdown():
        path_msg.header.stamp = rospy.Time.now()
        path_pub.publish(path_msg)
        rate.sleep()
