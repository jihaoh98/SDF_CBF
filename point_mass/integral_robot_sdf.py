import numpy as np
import sympy as sp
import yaml
import torch
from math import sqrt, cos, sin
from sympy.utilities.lambdify import lambdify
from cdf import CDF2D
from primitives2D_torch import Circle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Integral_Robot_Sdf:
    def __init__(self, params) -> None:
        """
        robot state: x, y, theta (only for plot, with constant value zero)
        controls: vx, vy
        circle_obstacle_state: ox, oy, ovx, ovy, o_radius
        define the clf and cbf based on point mass robot and circular-shaped obstacle
        """
        # robot system states, half width with half height
        self.state_dim = 3
        self.control_dim = 2
        self.margin = params['margin']

        # robot's current state
        x, y, theta = sp.symbols('x y theta')
        self.robot_state = sp.Matrix([x, y, theta])
        radius = sp.symbols('radius')
        self.robot_state_cbf = sp.Matrix([x, y, theta, radius])

        # robot's target state
        e_x, e_y, e_theta = sp.symbols('e_x e_y, e_theta')
        self.target_state = sp.Matrix([e_x, e_y, e_theta])

        # circular-shaped obstacle, 2D case
        self.cir_obs_dim = 5
        o_x, o_y, o_vx, o_vy, o_radius = sp.symbols('o_x o_y o_vx o_vy o_radius')
        self.cir_obstacle_state = sp.Matrix([o_x, o_y, o_vx, o_vy, o_radius])

        # add configuration distance field obstacle
        self.cdf_obs_dim = 5
        cdf_x, cdf_y, cdf_vx, cdf_vy, cdf_radius = sp.symbols('cdf_x cdf_y cdf_vx cdf_vy cdf_radius')
        self.cdf_obstacle_state = sp.Matrix([cdf_x, cdf_y, cdf_vx, cdf_vy, cdf_radius])

        # robot system dynamics
        self.f = None
        self.f_symbolic = None
        self.g = None
        self.g_symbolic = None

        # obstacle dynamics, single integral system
        self.cir_obstacle_dynamics = None
        self.cir_obstacle_dynamics_symbolic = None

        # cdf obstacle dynamics, single integral system
        self.cdf_obstacle_dynamics = None
        self.cdf_obstacle_dynamics_symbolic = None

        # clf design
        self.clf = None
        self.lf_clf = None
        self.lg_clf = None

        # cbf design
        self.cbf = None
        self.lf_cbf = None
        self.lg_cbf = None

        # initialize
        self.init_system()

    def init_system(self):
        """ init the system's dynamics and clf & cbf """
        # init the robot system dynamics and obstacle dynamics
        self.f_symbolic, self.g_symbolic = self.define_system_dynamics()
        self.f = lambdify([self.robot_state], self.f_symbolic)
        self.g = lambdify([self.robot_state], self.g_symbolic)

        # the circle obstacle dynamics
        self.cir_obstacle_dynamics_symbolic = sp.Matrix(
            [self.cir_obstacle_state[2], self.cir_obstacle_state[3], 0.0, 0.0, 0.0])
        self.cir_obstacle_dynamics = lambdify([self.cir_obstacle_state], self.cir_obstacle_dynamics_symbolic)

        # the cdf obstacle dynamics
        self.cdf_obstacle_dynamics_symbolic = sp.Matrix(
            [self.cdf_obstacle_state[2], self.cdf_obstacle_state[3], 0.0, 0.0])
        self.cdf_obstacle_dynamics = lambdify([self.cdf_obstacle_state], self.cdf_obstacle_dynamics_symbolic)

        self.init_clf()
        self.init_cbf()

    def define_system_dynamics(self):
        """ define the system dynamics """
        f = sp.Matrix([0, 0, 0])
        g = sp.Matrix([
            [1, 0],
            [0, 1],
            [0, 0]])

        return f, g

    def init_clf(self):
        """ init the control lyapunov function for navigation """
        H = sp.Matrix([[1.0, 0.0],
                       [0.0, 1.0]])
        relative_x = self.robot_state[0] - self.target_state[0]  # x - e_x
        relative_y = self.robot_state[1] - self.target_state[1]  # y - e_y
        relative_state = sp.Matrix([relative_x, relative_y])  # [x - e_x, y - e_y]

        clf_symbolic = (relative_state.T @ H @ relative_state)[0, 0]  # (x - e_x)^2 + (y - e_y)^2
        self.clf = lambdify([self.robot_state, self.target_state], clf_symbolic)  # clf(x, y, e_x, e_y)

        lf_clf_symbolic, lg_clf_symbolic = self.define_clf_derivative(clf_symbolic)  # lf_clf, lg_clf
        self.lf_clf = lambdify([self.robot_state, self.target_state], lf_clf_symbolic)  # lf_clf(x, y, e_x, e_y)
        self.lg_clf = lambdify([self.robot_state, self.target_state], lg_clf_symbolic)  # lg_clf(x, y, e_x, e_y)

    def define_clf_derivative(self, clf_symbolic):
        """ return the symbolic expression of lf_clf and lg_clf"""
        dx_clf_symbolic = sp.Matrix([clf_symbolic]).jacobian(self.robot_state)
        lf_clf = (dx_clf_symbolic @ self.f_symbolic)[0, 0]
        lg_clf = dx_clf_symbolic @ self.g_symbolic

        return lf_clf, lg_clf

    def init_cbf(self):
        """
        init the cbf between a point mass and a circular-shaped obstacle
        Args:
        point mass params: center_pose (x, y, theta), radius for visualization
        circle obs params: state of the circular obstacle (ox, oy, ovx, ovy, o_radius) 
        Returns:
            cbf based on Euclidean distance
        """

        # init the circle obstacle cbf based on Euclidean distance
        cbf_symbolic = (self.robot_state_cbf[0] - self.cir_obstacle_state[0]) ** 2 + (
                self.robot_state_cbf[1] - self.cir_obstacle_state[1]) ** 2
        cbf_symbolic = cbf_symbolic - (self.robot_state_cbf[3] + self.cir_obstacle_state[4]) ** 2
        cbf_symbolic = cbf_symbolic - self.margin

        self.cbf = lambdify([self.robot_state_cbf, self.cir_obstacle_state], cbf_symbolic)

        lf_cbf_symbolic, lg_cbf_symbolic, dt_cbf_symbolic = self.define_cbf_derivative(cbf_symbolic)
        self.lf_cbf = lambdify([self.robot_state_cbf, self.cir_obstacle_state], lf_cbf_symbolic)
        self.lg_cbf = lambdify([self.robot_state_cbf, self.cir_obstacle_state], lg_cbf_symbolic)
        self.dt_cbf = lambdify([self.robot_state_cbf, self.cir_obstacle_state], dt_cbf_symbolic)

    def define_cbf_derivative(self, cbf_symbolic):
        """ return the symbolic expression of lf_cbf, lg_cbf and dt_cbf """
        dx_cbf_symbolic = sp.Matrix([cbf_symbolic]).jacobian(self.robot_state)
        lf_cbf = (dx_cbf_symbolic @ self.f_symbolic)[0, 0]
        lg_cbf = dx_cbf_symbolic @ self.g_symbolic

        dox_cbf_symbolic = sp.Matrix([cbf_symbolic]).jacobian(self.cir_obstacle_state)

        dt_cbf = (dox_cbf_symbolic @ self.cir_obstacle_dynamics_symbolic)[0, 0]

        return lf_cbf, lg_cbf, dt_cbf

    def derive_cbf_gradient(self, robot_cur_state, obs_cur_state, dist_input, grad_input):
        dh_dxb = grad_input.flatten()

        # lf_cbf, lg_cbf, dt_obs_cbf (dynamic obstacle)
        lf_cbf, lg_cbf, dt_obs_cbf = self.get_cbf_gradient(robot_cur_state, obs_cur_state, dh_dxb)

        return lf_cbf, lg_cbf, dt_obs_cbf

    def get_cbf_gradient(self, robot_state, obstacle_state, cdf_gradient):
        """
        Args:
            robot_state: x, y, theta
            obstacle_state: ox, oy, ovx, ovy
            cir_obstacle_state: ox, oy, ovx, ovy, o_radius
            sdf_gradient: dh / dx_b
        Returns:
            lf_cbf in shape ()
            lg_cbf in shape (1, 2)
        """
        # h is calculated based on the relative position of obstacle
        # dh / dt = (dh / dx) * (dx / dt), x is a vector and dx / dt = f(x) + g(x) * u
        # x_b is the obstacle state in robot frame, x_o is the obstacle state in world frame, xr is the robot state
        # x_b = (x_o - xr)
        # dh / dx = [dh / dp, dh / dtheta = 0] p is xr (x, y)

        dh_dp = cdf_gradient
        dh_dx = np.array([dh_dp[0], dh_dp[1], 0])
        lf_cbf = (dh_dx @ self.f(robot_state))[0]
        lg_cbf = (dh_dx @ self.g(robot_state)).reshape(1, 2)

        # TODO: need to fix when considering the dynamic obstacle
        dt_obs_cbf = (cdf_gradient @ self.cdf_obstacle_dynamics(obstacle_state)[0:2])[0]

        return lf_cbf, lg_cbf, dt_obs_cbf

    def next_state(self, current_state, u, dt):
        """ simple one step """
        next_state = current_state
        next_state = next_state + dt * (
                self.f(current_state).T[0] + (self.g(current_state) @ np.array(u).reshape(self.control_dim, -1)).T[
            0])

        return next_state
