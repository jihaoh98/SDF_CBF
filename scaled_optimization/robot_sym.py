import numpy as np
import sympy as sp
import yaml
from math import sqrt, cos, sin
from sympy.utilities.lambdify import lambdify


class Integral_Robot_Sym:
    def __init__(self) -> None:
        """
        robot state: x, y, theta (only for plot, with constant value zero)
        controls: vx, vy
        define the clf function
        """
        # robot system states, half width with half height
        self.state_dim = 3
        self.control_dim = 2

        # robot's current state and target state
        x, y, theta = sp.symbols('x y theta')
        self.robot_state = sp.Matrix([x, y, theta])
        e_x, e_y, e_theta = sp.symbols('e_x e_y, e_theta')
        self.target_state = sp.Matrix([e_x, e_y, e_theta])

        # robot system dynamics
        self.f = None
        self.f_symbolic = None
        self.g = None
        self.g_symbolic = None

        # clf and cbf design
        self.clf = None
        self.clf_symbolic = None
        self.lf_clf = None
        self.lf_clf_symbolic = None
        self.lg_clf = None
        self.lg_clf_symbolic = None

        # initialize
        self.init_system()

    def init_system(self):
        """ init the system's dynamics and clf & cbf """
        # init the robot system dynamics and obstacle dynamics
        self.f_symbolic, self.g_symbolic = self.define_system_dynamics()
        self.f = lambdify([self.robot_state], self.f_symbolic)
        self.g = lambdify([self.robot_state], self.g_symbolic)

        self.init_clf()

    def define_system_dynamics(self):
        """ define the system dynamics """
        f = sp.Matrix([0, 0, 0])
        g = sp.Matrix([
            [1, 0], 
            [0, 1], 
            [0, 0]
        ])
        
        return f, g
    
    def init_clf(self):
        """ init the control lyapunov function for navigation """
        H = sp.Matrix(
            [[1.0, 0.0],
            [0.0, 1.0]]
        )
        relative_x = self.robot_state[0] - self.target_state[0]
        relative_y = self.robot_state[1] - self.target_state[1]
        relative_state = sp.Matrix([relative_x, relative_y])

        self.clf_symbolic = (relative_state.T @ H @ relative_state)[0, 0]
        self.clf = lambdify([self.robot_state, self.target_state], self.clf_symbolic)

        self.lf_clf_symbolic, self.lg_clf_symbolic = self.define_clf_derivative(self.clf_symbolic)
        self.lf_clf = lambdify([self.robot_state, self.target_state], self.lf_clf_symbolic)
        self.lg_clf = lambdify([self.robot_state, self.target_state], self.lg_clf_symbolic)
        
    def define_clf_derivative(self, clf_symbolic):
        """ return the symbolic expression of lf_clf and lg_clf"""
        dx_clf_symbolic = sp.Matrix([clf_symbolic]).jacobian(self.robot_state)
        lf_clf = (dx_clf_symbolic @ self.f_symbolic)[0, 0]
        lg_clf = dx_clf_symbolic @ self.g_symbolic

        return lf_clf, lg_clf
    
    def next_state(self, current_state, u, dt):
        """ simple one step """
        next_state = current_state
        next_state = next_state + dt * (
            self.f(current_state).T[0] + (self.g(current_state) @ np.array(u).reshape(self.control_dim, -1)).T[0]
        )

        return next_state


class Unicycle_Robot_Sym:
    def __init__(self) -> None:
        """
        robot state: x, y, theta
        controls: v, w
        define the clf function
        """
        # robot system states, half width with half height
        self.state_dim = 3
        self.control_dim = 2

        # robot's current state and target state
        x, y, theta = sp.symbols('x y theta')
        self.robot_state = sp.Matrix([x, y, theta])
        e_x, e_y, e_theta = sp.symbols('e_x e_y, e_theta')
        self.target_state = sp.Matrix([e_x, e_y, e_theta])

        # robot system dynamics
        self.f = None
        self.f_symbolic = None
        self.g = None
        self.g_symbolic = None

        # clf design
        # for distance between the current postion and goal position
        self.clf1 = None
        self.clf1_symbolic = None

        self.lf_clf1 = None
        self.lf_clf1_symbolic = None
        self.lg_clf1 = None
        self.lg_clf1_symbolic = None

        # for orientation
        self.clf2 = None
        self.clf2_symbolic = None

        self.lf_clf2 = None
        self.lf_clf2_symbolic = None
        self.lg_clf2 = None
        self.lg_clf2_symbolic = None

        # initialize
        self.init_system()

    def init_system(self):
        """ init the system's dynamics and clf & cbf """
        # init the robot system dynamics and obstacle dynamics
        self.f_symbolic, self.g_symbolic = self.define_system_dynamics()
        self.f = lambdify([self.robot_state], self.f_symbolic)
        self.g = lambdify([self.robot_state], self.g_symbolic)

        self.init_clf()

    def define_system_dynamics(self):
        """ define the system dynamics """
        f = sp.Matrix([0, 0, 0])
        g = sp.Matrix([
            [sp.cos(self.robot_state[2]), 0],
            [sp.sin(self.robot_state[2]), 0],
            [0, 1]
        ])

        return f, g

    def init_clf(self):
        """ init the control lyapunov functions for distance and orientation """
        self.init_clf1()
        self.init_clf2()

    def init_clf1(self):
        """ 
        navigate the robot to its destination 
        v(x) = (x_g - x) ** 2 + (y_g - y) ** 2
        """
        H = sp.Matrix([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        relative_x = self.robot_state[0] - self.target_state[0]
        relative_y = self.robot_state[1] - self.target_state[1]
        relative_state = sp.Matrix([relative_x, relative_y])

        self.clf1_symbolic = (relative_state.T @ H @ relative_state)[0, 0]
        self.clf1 = lambdify([self.robot_state, self.target_state], self.clf1_symbolic)

        self.lf_clf1_symbolic, self.lg_clf1_symbolic = self.define_clf_derivative(self.clf1_symbolic)
        self.lf_clf1 = lambdify([self.robot_state, self.target_state], self.lf_clf1_symbolic)
        self.lg_clf1 = lambdify([self.robot_state, self.target_state], self.lg_clf1_symbolic)

    def init_clf2(self):
        """ 
        adjust the orientation 
        v(x) = theta - arctan((y_g - y) / (x_g - x)) =>
        v(x) = (cos (theta) * (y_g - y) - sin (theta) * (x_g - x)) ** 2
        """
        dy = self.target_state[1] - self.robot_state[1]
        dx = self.target_state[0] - self.robot_state[0]
        self.clf2_symbolic = (sp.cos(self.robot_state[2]) * dy - sp.sin(self.robot_state[2]) * dx) ** 2
        self.clf2 = lambdify([self.robot_state, self.target_state], self.clf2_symbolic)

        self.lf_clf2_symbolic, self.lg_clf2_symbolic = self.define_clf_derivative(self.clf2_symbolic)
        self.lf_clf2 = lambdify([self.robot_state, self.target_state], self.lf_clf2_symbolic)
        self.lg_clf2 = lambdify([self.robot_state, self.target_state], self.lg_clf2_symbolic)

    def define_clf_derivative(self, clf_symbolic):
        """ return the symbolic expression of lf_clf and lg_clf"""
        dx_clf_symbolic = sp.Matrix([clf_symbolic]).jacobian(self.robot_state)
        lf_clf = (dx_clf_symbolic @ self.f_symbolic)[0, 0]
        lg_clf = dx_clf_symbolic @ self.g_symbolic

        return lf_clf, lg_clf

    def next_state(self, current_state, u, dt):
        """ simple one step """
        next_state = current_state
        next_state = next_state + dt * (
            self.f(current_state).T[0] + (self.g(current_state) @ np.array(u).reshape(self.control_dim, -1)).T[0]
        )

        return next_state


