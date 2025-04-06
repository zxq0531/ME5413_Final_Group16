#!/usr/bin/env python3
import casadi as ca
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.interpolate import splprep, splev

# Global variable for the true state
true_state = np.array([0.0, 0.0, 0.0])
# Global variable for the current smoothed path (Nx3 array: [x, y, theta])
current_smoothed_path = None
# Global variable for current raw waypoints (for visualization)
current_waypoints = None

def smooth_path(waypoints, num_points=200):
    """
    Smooth the trajectory given scatter waypoints using spline interpolation.
    Input:
      waypoints: numpy array of shape (M, 2) representing the raw path.
      num_points: number of points for the smooth path.
    Returns:
      smooth_traj: numpy array of shape (num_points, 3) containing [x, y, theta].
    """
    tck, u = splprep([waypoints[:, 0], waypoints[:, 1]], s=0)
    u_fine = np.linspace(0, 1, num_points)
    x_fine, y_fine = splev(u_fine, tck)
    dx = np.gradient(x_fine)
    dy = np.gradient(y_fine)
    theta_fine = np.arctan2(dy, dx)
    smooth_traj = np.column_stack((x_fine, y_fine, theta_fine))
    return smooth_traj

def generate_second_trajectory(previous_smooth_traj, num_waypoints=8, additional_length=15, y_deviation=0.5):
    """
    Generate a second trajectory that is very close to the first trajectory.
    The y-deviation is kept small.
    """
    last_point = previous_smooth_traj[-1, :]
    x_points = np.linspace(last_point[0], last_point[0] + additional_length, num_waypoints)
    y_points = last_point[1] + np.random.uniform(-y_deviation, y_deviation, num_waypoints)
    new_waypoints = np.vstack([x_points, y_points]).T
    new_smooth_traj = smooth_path(new_waypoints, num_points=200)
    return new_smooth_traj, new_waypoints

def plan_two_segments():
    """
    Plan two path segments that are close to each other and return a combined trajectory.
    Returns:
      combined_smooth_traj: Combined smooth trajectory.
      combined_waypoints: Combined raw waypoints.
    """
    num_waypoints_1 = 10
    x_points_1 = np.linspace(0, 20, num_waypoints_1)
    y_points_1 = np.random.uniform(-5, 5, num_waypoints_1)
    first_waypoints = np.vstack([x_points_1, y_points_1]).T
    first_smoothed_path = smooth_path(first_waypoints, num_points=200)
    
    second_smoothed_path, second_waypoints = generate_second_trajectory(first_smoothed_path,
                                                                        num_waypoints=8,
                                                                        additional_length=15,
                                                                        y_deviation=0.5)
    combined_smooth_traj = np.concatenate((first_smoothed_path, second_smoothed_path), axis=0)
    combined_waypoints = np.concatenate((first_waypoints, second_waypoints), axis=0)
    return combined_smooth_traj, combined_waypoints

def visualize(ax, smooth_traj, waypoints, ref_traj, vehicle_path, current_state, show_visual=True):
    """
    Visualize planned path, raw waypoints, reference trajectory, and vehicle path.
    """
    if not show_visual:
        return
    ax.clear()
    ax.plot(smooth_traj[:, 0], smooth_traj[:, 1], 'k--', label="Planned Path")
    ax.plot(waypoints[:, 0], waypoints[:, 1], 'ro', label="Waypoints")
    ax.plot(ref_traj[:, 0], ref_traj[:, 1], 'r-', label="Reference Trajectory")
    vehicle_array = np.array(vehicle_path)
    ax.plot(vehicle_array[:, 0], vehicle_array[:, 1], 'b-', label="Vehicle Path")
    car_size = 0.3
    c = current_state[0:2]
    theta_car = current_state[2]
    corners = np.array([[-car_size/2, -car_size/2],
                        [ car_size/2, -car_size/2],
                        [ car_size/2,  car_size/2],
                        [-car_size/2,  car_size/2]])
    R_mat = np.array([[np.cos(theta_car), -np.sin(theta_car)],
                      [np.sin(theta_car),  np.cos(theta_car)]])
    rotated_corners = (R_mat @ corners.T).T + c
    car_patch = Polygon(rotated_corners, closed=True, color='g', label="Vehicle")
    ax.add_patch(car_patch)
    ax.set_xlim(-5, 30)
    ax.set_ylim(-15, 15)
    ax.set_title("NMPC Tracking with Pre-planned Two-Segment Path")
    ax.legend()
    plt.pause(0.001)

class AutonomousCarNMPC:
    def __init__(self, T=0.1, N=10, param=None):
        if param is None:
            param = {}
        self.T = T
        self.N = N
        self.state_dim = 3
        self.control_dim = 2

        self.Q = param.get("Q", np.diag([5, 5, 0]))
        self.R = param.get("R", np.diag([0.2, 0.2]))
        self.Qf = param.get("Qf", None) or self.Q
        self.initial_state = param.get("initial_state", np.array([0.0, 0.0, 0.0]))
        self.kalman_P0 = param.get("kalman_P0", None)
        self.kalman_Q = param.get("kalman_Q", np.eye(3) * 0.01)
        self.kalman_R = param.get("kalman_R", np.eye(3) * 0.1)
        self.use_kalman_filter = param.get("use_kalman_filter", True)
        self.max_lin_acc = param.get("max_lin_acc", 2.0)
        self.max_ang_acc = param.get("max_ang_acc", 2.0)
        self.max_lin_vel = param.get("max_lin_vel", 2.0)
        self.max_ang_vel = param.get("max_ang_vel", 2.0)
        self.vel_ref = param.get("vel_ref", 1.0)
        self.vel_penalty = param.get("vel_penalty", 10)

        self.opti = ca.Opti()
        self.X = self.opti.variable(self.N+1, self.state_dim)
        self.U = self.opti.variable(self.N, self.control_dim)
        self.X0 = self.opti.parameter(1, self.state_dim)
        self.X_ref = self.opti.parameter(self.N+1, self.state_dim)
        self.U0 = self.opti.parameter(1, self.control_dim)

        def f(x, u):
            return ca.vertcat(u[0]*ca.cos(x[2]),
                              u[0]*ca.sin(x[2]),
                              u[1])
        for k in range(self.N):
            x_next = self.X[k, :] + self.T * f(self.X[k, :], self.U[k, :]).T
            self.opti.subject_to(self.X[k+1, :] == x_next)
        self.opti.subject_to(self.X[0, :] == self.X0)

        cost = 0
        for k in range(self.N):
            state_error = self.X[k, :] - self.X_ref[k, :]
            control = self.U[k, :]
            cost += ca.mtimes([state_error, self.Q, state_error.T]) + ca.mtimes([control, self.R, control.T])
            cost += self.vel_penalty * (self.U[k,0] - self.vel_ref)**2
        terminal_error = self.X[self.N, :] - self.X_ref[self.N, :]
        cost += ca.mtimes([terminal_error, self.Qf, terminal_error.T])
        self.opti.minimize(cost)

        v_max = self.max_lin_vel
        omega_max = self.max_ang_vel
        for k in range(self.N):
            self.opti.subject_to(self.opti.bounded(-v_max, self.U[k,0], v_max))
            self.opti.subject_to(self.opti.bounded(-omega_max, self.U[k,1], omega_max))
        self.opti.subject_to(self.opti.bounded(-self.max_lin_acc*self.T, self.U[0,0]-self.U0[0,0], self.max_lin_acc*self.T))
        self.opti.subject_to(self.opti.bounded(-self.max_ang_acc*self.T, self.U[0,1]-self.U0[0,1], self.max_ang_acc*self.T))
        for k in range(self.N-1):
            self.opti.subject_to(self.opti.bounded(-self.max_lin_acc*self.T, self.U[k+1,0]-self.U[k,0], self.max_lin_acc*self.T))
            self.opti.subject_to(self.opti.bounded(-self.max_ang_acc*self.T, self.U[k+1,1]-self.U[k,1], self.max_ang_acc*self.T))

        opts = {"ipopt.print_level": 0, "print_time": 0}
        self.opti.solver("ipopt", opts)

        self.X_opt = np.tile(self.initial_state, (self.N+1, 1))
        self.U_opt = np.zeros((self.N, self.control_dim))

        self.state_est = self.initial_state.copy()
        self.P = np.eye(self.state_dim) if self.kalman_P0 is None else self.kalman_P0

    def kalman_update(self, u, z):
        theta = self.state_est[2]
        A = np.array([[1, 0, -self.T*u[0]*np.sin(theta)],
                      [0, 1,  self.T*u[0]*np.cos(theta)],
                      [0, 0, 1]])
        x_pred = self.state_est + self.T * np.array([u[0]*np.cos(theta),
                                                     u[0]*np.sin(theta),
                                                     u[1]])
        P_pred = A @ self.P @ A.T + self.kalman_Q
        H = np.eye(self.state_dim)
        K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + self.kalman_R)
        self.state_est = x_pred + K @ (z - x_pred)
        self.P = (np.eye(self.state_dim) - K @ H) @ P_pred
        return self.state_est

    def solve_nmpc(self, current_state, ref_traj, prev_u=None):
        current_state = current_state.reshape(1, -1)
        self.opti.set_value(self.X0, current_state)
        self.opti.set_value(self.X_ref, ref_traj)
        if prev_u is None:
            prev_u = np.zeros(self.control_dim)
        else:
            prev_u = np.array(prev_u).reshape(1, self.control_dim)
        self.opti.set_value(self.U0, prev_u)
        self.opti.set_initial(self.X, self.X_opt)
        self.opti.set_initial(self.U, self.U_opt)
        try:
            sol = self.opti.solve()
        except Exception as e:
            print("NMPC solve failed:", e)
            return np.zeros(self.control_dim)
        self.X_opt = sol.value(self.X)
        self.U_opt = sol.value(self.U)
        return self.U_opt[0, :]

    def control_loop(self, ref_traj_generator, measurement_function, control_frequency=10, iterations=200, show_visual=True):
        dt = 1.0/control_frequency
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        fig, ax = plt.subplots()
        vehicle_trajectory = []
        global true_state, current_smoothed_path, current_waypoints
        for i in range(iterations):
            z = measurement_function()
            current_u = self.U_opt[0, :] if self.U_opt.size else np.zeros(self.control_dim)
            if self.use_kalman_filter:
                current_state = self.kalman_update(current_u, z)
            else:
                current_state = z

            ref_traj = ref_traj_generator(current_state, self.N+1, self.T)
            u_opt = self.solve_nmpc(current_state, ref_traj, current_u)
            print("Iteration {}: Control command: v = {:.2f} m/s, omega = {:.2f} rad/s".format(i, u_opt[0], u_opt[1]))
            
            theta = true_state[2]
            true_state = true_state + self.T*np.array([u_opt[0]*np.cos(theta),
                                                       u_opt[0]*np.sin(theta),
                                                       u_opt[1]])
            vehicle_trajectory.append(true_state.copy())
            visualize(ax, current_smoothed_path, current_waypoints, ref_traj, vehicle_trajectory, true_state, show_visual)
            time.sleep(dt)

def ref_traj_generator_path(current_state, horizon, T):
    global current_smoothed_path
    distances = np.sqrt((current_smoothed_path[:,0]-current_state[0])**2+
                        (current_smoothed_path[:,1]-current_state[1])**2)
    idx = np.argmin(distances)
    ref_traj = np.zeros((horizon,3))
    for i in range(horizon):
        index = idx+i
        if index>=len(current_smoothed_path):
            index = len(current_smoothed_path)-1
        ref_traj[i,:] = current_smoothed_path[index,:]
    return ref_traj

def measurement_function():
    global true_state
    noise = np.random.randn(3)*0.05
    return true_state + noise

if __name__ == '__main__':
    # Plan two segments at the beginning.
    current_smoothed_path, current_waypoints = plan_two_segments()
    print("Planned a combined two-segment path.")
    
    params = {
        "Q": np.diag([5, 5, 1]),
        "R": np.diag([0.2, 0.2]),
        "initial_state": np.array([0.0, 0.0, 0.0]),
        "max_lin_acc": 2.0,
        "max_ang_acc": 2.0,
        "max_lin_vel": 2.0,
        "max_ang_vel": 2.0,
        "vel_ref": 1.0,
        "vel_penalty": 0
    }
    nmpc_controller = AutonomousCarNMPC(T=0.1, N=10, param=params)

    nmpc_controller.control_loop(ref_traj_generator_path, measurement_function,
                                 control_frequency=10, iterations=500, show_visual=True)
