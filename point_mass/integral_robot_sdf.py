import numpy as np
import sympy as sp
import yaml
from math import sqrt, cos, sin
from sympy.utilities.lambdify import lambdify


class Integral_Robot_Sdf:
    def __init__(self, params) -> None:
        """
        robot state: x, y, z
        controls: vx, vy, vz
        circle_obstacle_state: ox, oy, oz, ovx, ovy, ovz, o_radius
        define the clf and cbf based on point mass robot and circular-shaped obstacle in 3D
        """
        # robot system states, half width with half height
        self.state_dim = 3
        self.control_dim = 3
        self.margin = params['margin']

        # robot's current state
        x, y, z = sp.symbols('x y z')
        self.robot_state = sp.Matrix([x, y, z])
        radius = sp.symbols('radius')
        self.robot_state_cbf = sp.Matrix([x, y, z, radius])

        # robot's target state
        e_x, e_y, e_z = sp.symbols('e_x e_y e_z')
        self.target_state = sp.Matrix([e_x, e_y, e_z])

        # circular-shaped obstacle
        self.cir_obs_dim = 7
        o_x, o_y, o_z, o_vx, o_vy, o_vz, o_radius = sp.symbols('o_x o_y o_z o_vx o_vy o_vz o_radius')
        self.cir_obstacle_state = sp.Matrix([o_x, o_y, o_z, o_vx, o_vy, o_vz, o_radius])

        # robot system dynamics
        self.f = None
        self.f_symbolic = None
        self.g = None
        self.g_symbolic = None

        # obstacle dynamics, single integral system
        self.cir_obstacle_dynamics = None
        self.cir_obstacle_dynamics_symbolic = None

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

        self.cir_obstacle_dynamics_symbolic = sp.Matrix(
            [
                self.cir_obstacle_state[3],
                self.cir_obstacle_state[4],
                self.cir_obstacle_state[5],
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        self.cir_obstacle_dynamics = lambdify([self.cir_obstacle_state], self.cir_obstacle_dynamics_symbolic)

        self.init_clf()
        self.init_cbf()

    def define_system_dynamics(self):
        """ define the system dynamics """
        f = sp.Matrix([0, 0, 0])
        g = sp.Matrix([
            [1, 0, 0], 
            [0, 1, 0], 
            [0, 0, 1]])

        return f, g

    def init_clf(self):
        """ init the control lyapunov function for navigation """
        H = sp.Matrix([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])
        relative_x = self.robot_state[0] - self.target_state[0]
        relative_y = self.robot_state[1] - self.target_state[1]
        relative_z = self.robot_state[2] - self.target_state[2]
        relative_state = sp.Matrix([relative_x, relative_y, relative_z])

        clf_symbolic = (relative_state.T @ H @ relative_state)[0, 0]
        self.clf = lambdify([self.robot_state, self.target_state], clf_symbolic)

        lf_clf_symbolic, lg_clf_symbolic = self.define_clf_derivative(clf_symbolic)
        self.lf_clf = lambdify([self.robot_state, self.target_state], lf_clf_symbolic)
        self.lg_clf = lambdify([self.robot_state, self.target_state], lg_clf_symbolic)

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
        point mass params: center_pose (x, y, z), radius for visualization
        circle obs params: state of the circular obstacle (ox, oy, oz, ovx, ovy, ovz, o_radius) 
        Returns:
            cbf based on Euclidean distance
        """
        cbf_symbolic = (
            (self.robot_state_cbf[0] - self.cir_obstacle_state[0]) ** 2
            + (self.robot_state_cbf[1] - self.cir_obstacle_state[1]) ** 2
            + (self.robot_state_cbf[2] - self.cir_obstacle_state[2]) ** 2
        )
        cbf_symbolic = (
            cbf_symbolic - (self.robot_state_cbf[3] + self.cir_obstacle_state[6]) ** 2
        )
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

    def next_state(self, current_state, u, dt):
        """ simple one step """
        next_state = current_state
        next_state = next_state + dt * (self.f(current_state).T[0] + (self.g(current_state) @ np.array(u).reshape(self.control_dim, -1)).T[0])

        return next_state
