import numpy as np
import sympy as sp
import yaml
from sympy.utilities.lambdify import lambdify
from math import sqrt


class Robot_Sdf:
    def __init__(self, params) -> None:
        """
        robot_state: x y
        controls   : vx, vy
        obstacle_state: ox, oy, ovx, ovy
        define the clf and cbf based on sdf for rectangle-shaped robot withrespect to polytopic-shaped obstacles
        """
        # robot system states, width with height is half
        self.state_dim = 2
        self.control_dim = 2
        self.rec_width = params['width']
        self.rec_height = params['height']
        self.margin = params['margin']
        self.e0 = float(params['e0'])

        # robot current state
        x, y = sp.symbols('x y')
        self.robot_state = sp.Matrix([x, y])

        # target state
        e_x, e_y = sp.symbols('e_x e_y')
        self.target_state = sp.Matrix([e_x, e_y])

        # obstacle state
        o_x, o_y, o_vx, o_vy = sp.symbols('o_x o_y o_vx o_vy')
        self.obstacle_state = sp.Matrix([o_x, o_y, o_vx, o_vy])

        # robot system dynamics
        self.f = None
        self.f_symbolic = None
        self.g = None
        self.g_symbolic = None

        # obstacle dynamics, single integral system
        self.obstacle_dynamics = None
        self.obstacle_dynamics_symbolic = None

        # clf design
        self.clf = None
        self.clf_symbolic = None
        self.lf_clf = None
        self.lf_clf_symbolic = None
        self.lg_clf = None
        self.lg_clf_symbolic = None

        # cbf design, distance of a point to a rectangle
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
        init the cbf based on sdf to calculate the sdf between rectangle and a point from the obstacle
        Args:
        rectangle params: center, width, height (half of itself), in fact only position
        point params: state of a point in the edge of the obstacle ox, oy, ovx, ovy

        Returns:
            sdf between rectangle and point: float
        """
        dx = sp.Abs(self.obstacle_state[0] - self.robot_state[0]) - self.rec_width
        dy = sp.Abs(self.obstacle_state[1] - self.robot_state[1]) - self.rec_height

        distance_inside = sp.Min(sp.Max(dx, dy), 0.0)
        distance_outside = sp.sqrt(sp.Max(dx, 0) ** 2 + sp.Max(dy, 0) ** 2)
        distance = distance_outside + distance_inside

        # for point, without minus obstacle_radius
        # add margin for collision aviodance
        self.cbf_symbolic = distance - self.margin
        self.cbf = lambdify([self.robot_state, self.obstacle_state], self.cbf_symbolic)

    def get_cbf_value(self, relative_position):
        """ get cbf value based on the relative position """
        dx = abs(relative_position[0]) - self.rec_width
        dy = abs(relative_position[1]) - self.rec_height

        distance_inside = min(max(dx, dy), 0.0)
        distance_outside = sqrt(max(dx, 0) ** 2 + max(dy, 0) ** 2)
        distance = distance_outside + distance_inside

        # add margin for collision avoidance
        distance = distance - self.margin
        return distance

    def get_gradient_1(self, robot_state, obstacle_state):
        """ based on world frame """
        sdf = self.cbf(robot_state, obstacle_state)
        sdf_gradient = np.zeros((obstacle_state[0:2].shape[0], ))
        for i in range(obstacle_state[0:2].shape[0]):
            e = np.zeros((obstacle_state.shape[0], ))
            e[i] = self.e0
            sdf_next = self.cbf(robot_state, obstacle_state + e)
            sdf_gradient[i] = (sdf_next - sdf) / self.e0

        return sdf_gradient

    def get_gradient_2(self, robot_state, obstacle_state):
        """ based on robot frame """
        relative_position = obstacle_state[0:2] - robot_state
        sdf = self.get_cbf_value(relative_position)

        sdf_gradient = np.zeros((obstacle_state[0:2].shape[0], ))
        for i in range(obstacle_state[0:2].shape[0]):
            e = np.zeros((obstacle_state[0:2].shape[0], ))
            e[i] = self.e0
            sdf_next = self.get_cbf_value(relative_position + e)
            sdf_gradient[i] = (sdf_next - sdf) / self.e0

        return sdf_gradient

    def derive_cbf_gradient(self, robot_state, obstacle_state):
        """
        derive the gradient of sdf between the rectangle robot and a point from the rectangle osbatcle
        dh / dt = (dh / dx) * (dx / dt)   x is a vector and dx / dt = f + g * u

        x_or = x_o - xr (x is xr, which means robot's state)
        dh / dx = (d x_or / dx) * (dh / d x_or)
        d x_or / dx = -I (unit matrix)

        dh / dx = -dh / d x_or
        dh / dt = (-dh / d x_or) * (dx / dt)
        This code calculate the gradient based on numerical values.

        Args:
            robot state: [x, y]
            obstacle_state: [ox, oy, ovx, ovy], only one point of polytopic-shaped obstacle

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

        # gradient for cbf with respect to dynamic obstacle
        # dh / dt = (dh / d_ox) * (d_ox / dt) 

        # x_or = x_o - x
        # dh / d_ox = (d x_or / d_ox) * (dh / d x_or)
        # d x_or / d_ox = I (unit matrix)

        # dh / d_ox = dh / d x_or
        # dh / dt = (dh / d x_or) * (d_ox / dt) 
        # Assume the velocity of obstacle is a constant
        dt_obs_cbf = (sdf_gradient @ self.obstacle_dynamics(obstacle_state)[0:2])[0]

        return lf_cbf, lg_cbf, dt_obs_cbf
    
    def get_sampled_points_from_obstacle_vertexes(self, obstacle_vertexes, num_samples):
        """
        get the sample points from the obstacle vertexes
        Params:
            obstacle vextexes: obstacle vertex posistion in anticlockwise (n, 2)
            num_samples: the points sampled in each edge

        Returns:
            sampled points in np.array (n, 2)
        """

        sample_points = []
        t = np.linspace(0, 1, num_samples)

        # sample points
        num_vertexes = obstacle_vertexes.shape[0]
        for i in range(num_vertexes):
            dx, dy = np.subtract(obstacle_vertexes[(i + 1) % num_vertexes], obstacle_vertexes[i]) 
            edge_points = [[obstacle_vertexes[i][0] + dx * tt, obstacle_vertexes[i][1] + dy * tt] for tt in t[:-1]]
            sample_points.extend(edge_points)

        return np.array(sample_points)
        
    def next_state(self, current_state, u, dt):
        """ simple one step """
        next_state = current_state
        next_state = next_state + dt * (self.f(current_state).T[0] + (self.g(current_state) @ np.array(u).reshape(self.control_dim, -1)).T[0])

        return next_state

if __name__ == '__main__':
    file_name = 'settings.yaml'
    with open(file_name) as file:
        config = yaml.safe_load(file)

    robot_params = config['robot']
    obs_params = config['obstacle_list']
    test_target = Robot_Sdf(robot_params)

    # init
    for i in np.arange(0, 2.0, 0.1):
        robot_state = np.array([0.5, i])
        # target_state = np.array([1.0, 2.0])
        # obstacle_vertex = np.array(obs_params['obs_vertexes'])
        # obstacle_vel = np.array(obs_params['obs_vel'])
        obstacle_state = np.array([1.5, 1.5, 0.1, 0.2])
        a = test_target.get_gradient_1(robot_state, obstacle_state)
        b = test_target.get_gradient_2(robot_state, obstacle_state)
        if a.all() != b.all():
            print(robot_state)

    # print(test_target.clf(robot_state, target_state))
    # print(test_target.lf_clf(robot_state, target_state))
    # print(test_target.lg_clf(robot_state, target_state))    

    # sampled_points = test_target.get_sampled_points_from_obstacle_vertexes(obstacle_vertex, 6)
    # for i in range(sampled_points.shape[0]):
    #     current_point_state = np.array([sampled_points[i][0], sampled_points[i][1], obstacle_vel[0], obstacle_vel[1]])
    #     print(current_point_state)
