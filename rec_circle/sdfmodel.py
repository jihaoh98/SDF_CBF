import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify


class Sdf_Model:
    def __init__(self, params) -> None:
        """
        robot_state: x y
        controls   : vx, vy
        obstacle_state: ox, oy
        define the clf and cbf based on sdf for robots
        """

        # system states
        self.state_dim = 2
        self.control_dim = 2
        self.rec_width = params['width']
        self.rec_height = params['height']
        self.margin = params['margin']
        self.e0 = params['e0']

        # robot state
        x, y = sp.symbols('x y')
        self.robot_state = sp.Matrix([x, y])

        # target state
        e_x, e_y = sp.symbols('e_x e_y')
        self.target_state = sp.Matrix([e_x, e_y])

        # obstacle state
        o_x, o_y, o_vx, o_vy = sp.symbols('o_x o_y o_vx o_vy')
        self.obstacle_state = sp.Matrix([o_x, o_y, o_vx, o_vy])
        self.obstacle_radius = params['obstacle_radius']

        # robot system dynamics
        self.f = None
        self.f_symbolic = None
        self.g = None
        self.g_symbolic = None

        # obstacle dynamics
        self.obstacle_dynamics = None
        self.obstacle_dynamics_symbolic = None

        # clf and cbf design
        self.clf = None
        self.clf_symbolic = None
        self.lf_clf = None
        self.lf_clf_symbolic = None
        self.lg_clf = None
        self.lg_clf_symbolic = None

        self.cbf = None
        self.cbf_symbolic = None

        # initialize
        self.init_system()

    def init_system(self):
        """ init the system dynamics and clf & cbf """

        # init the robot system dynamics and obstacle dynamics
        self.f_symbolic, self.g_symbolic = self.define_system_dynamics()
        self.f = lambdify([self.robot_state], self.f_symbolic)
        self.g = lambdify([self.robot_state], self.g_symbolic)

        self.obstacle_dynamics_symbolic = sp.Matrix([self.obstacle_state[2], self.obstacle_state[3], 0.0, 0.0])
        self.obstacle_dynamics = lambdify([self.obstacle_state], self.obstacle_dynamics_symbolic)

        self.init_clf()
        self.init_cbf()

    def define_system_dynamics(self):
        f = sp.Matrix([0, 0])
        g = sp.Matrix([[1, 0],
                       [0, 1]])
        
        return f, g
    
    def init_clf(self):
        H = sp.Matrix([[1.0, 0.0],
                       [0.0, 1.0]])
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

    def init_cbf(self):
        """
        init the cbf based on sdf to calculate the sdf between rectangle and circle
        Args:
        rectangle params: center, width, height (half of itself)
        circle    params: center, radius

        Returns:
            sdf between rectangle and circle: float
        """
        dx = sp.Abs(self.obstacle_state[0] - self.robot_state[0]) - self.rec_width
        dy = sp.Abs(self.obstacle_state[1] - self.robot_state[1]) - self.rec_height

        distance_inside = sp.Min(sp.Max(dx, dy), 0.0)
        distance_outside = sp.sqrt(sp.Max(dx, 0) ** 2 + sp.Max(dy, 0) ** 2)
        distance = distance_outside + distance_inside

        self.cbf_symbolic = distance - self.obstacle_radius
        # add margin for collision aviodance
        self.cbf_symbolic = self.cbf_symbolic - self.margin
        self.cbf = lambdify([self.robot_state, self.obstacle_state], self.cbf_symbolic)

    def derive_cbf_gradient(self, robot_state, obstacle_state):
        """
        dh / dt = (dh / dx) * (dx / dt) 
        x is a vector and dx / dt = f + g * u

        x_or = x_o - xr (x is xr)
        dh / dx = (d x_or / dx) * (dh / d x_or)
        d x_or / dx = -I (unit matrix)

        dh / dx = -dh / d x_or
        dh / dt = (-dh / d x_or) * (dx / dt)
        This code calculate the gradient based on numerical value.

        Args:
            robot state: [x, y]
            obstacle_state: [ox, oy, ovx, ovy]

        Returns:
            lf_cbf in shape ()
            lg_cbf in shape (1, 2)
            dt_obs_cbf in shape ()
        """
        sdf = self.cbf(robot_state, obstacle_state)
        sdf_gradient = np.zeros((obstacle_state[0:2].shape[0], ))
        for i in range(obstacle_state[0:2].shape[0]):
            e = np.zeros((obstacle_state.shape[0], ))
            e[i] = self.e0
            sdf_next = self.cbf(robot_state, obstacle_state + e)
            sdf_gradient[i] = (sdf_next - sdf) / self.e0

        # sdf_greadient is equal to dh / d x_or
        # dx_cbf is equal to dh / dx
        dx_cbf = - sdf_gradient

        # get lf_cbf and lg_cbf
        lf_cbf = (dx_cbf @ self.f(robot_state))[0]
        lg_cbf = (dx_cbf @ self.g(robot_state)).reshape(1, 2)

        # gradient for cbf withrespect to dynamic obstacle
        # dh / dt = (dh / d_ox) * (d_ox / dt) 

        # x_or = x_o - x
        # dh / d_ox = (d x_or / d_ox) * (dh / d x_or)
        # d x_or / d_ox = I (unit matrix)

        # dh / d_ox = dh / d x_or
        # dh / dt = (dh / d x_or) * (d_ox / dt) 
        dt_obs_cbf = (sdf_gradient @ self.obstacle_dynamics(obstacle_state)[0:2])[0]

        return lf_cbf, lg_cbf, dt_obs_cbf
    
    def next_state(self, current_state, u, dt):
        """ simple one step """
        next_state = current_state
        next_state = next_state + dt * (self.f(current_state).T[0] + (self.g(current_state) @ np.array(u).reshape(self.control_dim, -1)).T[0])

        return next_state

if __name__ == '__main__':
    parameters = {'width': 1.0, 'height': 0.5, 'obstacle_radius': 0.5, 'e0': 1E-6, 'margin': 0.6}
    test_target = Sdf_Model(params=parameters)

    robot_state = np.array([0.5, 1.0])
    target_state = np.array([1.0, 2.0])
    obstacle_state = np.array([2.0, 3.0, 0.0, 0.0])
    print(test_target.clf(robot_state, target_state))
    print(test_target.lf_clf(robot_state, target_state))
    print(test_target.lg_clf(robot_state, target_state))
    # print(test_target.derive_cbf_gradient(robot_state, obstacle_state))
    # for i in np.arange(1.0, 2.0, 0.1):
    #     for j in np.arange(1.5, 3.0, 0.1):
    #         obstacle_state = np.array([i, j, 0.0, 0.0])
    #         pass
    #         # sdf1 = calculate_sdf_rectangle_circle(robot_state, parameters['width'], parameters['height'], obstacle_state[0:2], parameters['obstacle_radius'])
    #         # sdf2 = test_target.cbf(robot_state, obstacle_state)
    #         # if sdf1 == sdf2:
    #         #     print(i, j)

    #         # grad1 = calculate_sdf_gradient_with_obstacle(robot_state, parameters['width'], parameters['height'], obstacle_state[0:2], parameters['obstacle_radius'])
    #         # grad2 = test_target.derive_cbf_gradient(robot_state, obstacle_state)
    #         # if grad1.all() == grad2.all():
    #         #     print(i, j)




