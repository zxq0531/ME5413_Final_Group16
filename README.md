# ME5413 Final Project 


This project is the final project for ME5413 and integrates mapping, global path planning, local path planning, and control for a mobile robot. The overall system is divided into the following modules:

- **Mapping:** Constructs the environment map.
- **Global Path Planning:** Uses an A* algorithm combined with a "cow plowing" method to generate a global path.
- **Local Path Planning:** Implements the Dynamic Window Approach (DWA) to generate a smooth, feasible local trajectory.
- **Controller:** Uses Nonlinear Model Predictive Control (NMPC) to track the planned trajectory and control the vehicle.

## Features

- **Mapping:** Create a map of the environment using ROS map_server.
- **Global Path Planning:** Generate an optimal path from start to goal using A* enhanced with a "cow plowing" algorithm.
- **Local Path Planning:** Refine the global path with DWA to account for dynamic obstacles and vehicle kinematics.
- **NMPC Controller:** Track the local path with NMPC to provide smooth and robust control commands.
- **One-Click Startup:** All modules are integrated to start with a single command.

## Mapping

### Dependencies
* SLAM requirements(Refer to different documents based on the method you choose.)
  * [Fast-lio](https://github.com/hku-mars/FAST_LIO)
  * [Cartographer](https://google-cartographer-ros.readthedocs.io/en/latest/)
  * GMapping (should be available by default in the project)

### Usage

### 0. Gazebo World

This command will launch the gazebo with the project world

```bash
# Launch Gazebo World together with our robot
roslaunch me5413_world world.launch
```

### 1. Mapping

After launching **Step 0**, in the second terminal:

```bash
# Launch GMapping
roslaunch me5413_world mapping.launch

# or launch Cartographer 2D
roslaunch final_slam cartographer_2d.launch

# or launch Cartographer 3D
roslaunch final_slam cartographer_3d.launch

# or launch Fast-lio
roslaunch final_slam fast_lio.launch
```

After finishing GMapping and Cartographer mapping, run the following command in the thrid terminal to save the map:

```bash
# Save the map as `my_map` in the `maps/` folder
roscd final_slam/maps/
rosrun map_server map_saver -f my_map map:=/map
```

Saving maps with Fast-lio is relatively complex; please refer to [Fast-lio](https://github.com/hku-mars/FAST_LIO) for guidance.
Finally, the map we used is shown below.
![rviz_nmapping_image](./maps/my_map.png)

## Installation

1. Clone the repository into your ROS workspace:
    ```bash
    cd ~/ME5413_Final_Group16
    ```
2. Build the project using `catkin_make`:
    ```bash
    catkin_make
    ```
3. Source the workspace:
    ```bash
    source devel/setup.bash
    ```

## Usage

To start the system (including global and local path planning as well as vehicle control), simply run:
```bash
roslaunch pnc start.launch
```

