import numpy as np
import sympy as sp
import yaml
from math import sqrt, cos, sin
from sympy.utilities.lambdify import lambdify


class Robot_Sdf:
    def __init__(self, params) -> None:
        """
        robot state: x, y, theta
        controls: v, w
        obstacle_state: ox, oy, ovx, ovy
        define the clf and cbf based on sdf for rectangle-shaped robot with respect to any-shaped obstacles (sampled in the edges)
        """
        # robot system states, half width with half height
        self.state_dim = 3
        self.control_dim = 2
        self.rec_width = params['width']
        self.rec_height = params['height']
        self.margin = params['margin']
        self.e0 = float(params['e0'])

        # robot current state
        x, y, theta = sp.symbols('x y theta')
        self.robot_state = sp.Matrix([x, y, theta])

        # target state
        e_x, e_y, e_theta = sp.symbols('e_x e_y, e_theta')
        self.target_state = sp.Matrix([e_x, e_y, e_theta])

        # obstacle state
        self.obs_dim = 4
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
        # for distance between current postion and goal position
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

        # constant velocity
        self.obstacle_dynamics_symbolic = sp.Matrix([self.obstacle_state[2], self.obstacle_state[3], 0.0, 0.0])
        self.obstacle_dynamics = lambdify([self.obstacle_state], self.obstacle_dynamics_symbolic)

        self.init_clf()
        self.init_cbf()

    def define_system_dynamics(self):
        """ define the system dynamics """
        f = sp.Matrix([0, 0, 0])
        g = sp.Matrix([
            [sp.cos(self.robot_state[2]), 0], 
            [sp.sin(self.robot_state[2]), 0], 
            [0, 1]])
        
        return f, g
    
    def init_clf(self):
        """ init the control lyapunov function """
        self.init_clf1()
        self.init_clf2()
        
    def init_clf1(self):
        """ 
        navigate the robot to its destination 
        v(x) = (x_g - x) ** 2 + (y_g - y) ** 2
        """
        H = sp.Matrix([[1.0, 0.0],
                       [0.0, 1.0]])
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
        self.clf2_symbolic = (sp.cos(self.robot_state[2]) * (self.target_state[1] - self.robot_state[1]) - sp.sin(self.robot_state[2]) * (self.target_state[0] - self.robot_state[0])) ** 2
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

    def init_cbf(self):
        """
        init the cbf based on sdf to calculate the sdf between rectangle and a point from the obstacle's edge
        Args:
        rectangle params: center_pose (x, y, theta), half width and height
        point params: state of a point in the edge of the obstacle (ox, oy, ovx, ovy) (in world frame)
        Returns:
            sdf between rectangle and point: float
        """

        R_inverse = sp.Matrix([[sp.cos(self.robot_state[2]), sp.sin(self.robot_state[2])],
                               [-sp.sin(self.robot_state[2]), sp.cos(self.robot_state[2])]])
        
        relative_position = sp.Matrix([[self.obstacle_state[0] - self.robot_state[0]],
                                       [self.obstacle_state[1] - self.robot_state[1]]])
        
        # get position in the robot frame
        relative_position = R_inverse @ relative_position
        dx = sp.Abs(relative_position[0]) - self.rec_width
        dy = sp.Abs(relative_position[1]) - self.rec_height

        distance_inside = sp.Min(sp.Max(dx, dy), 0.0)
        distance_outside = sp.sqrt(sp.Max(dx, 0) ** 2 + sp.Max(dy, 0) ** 2)
        distance = distance_outside + distance_inside

        # for point, without minus obstacle_radius
        # add margin for collision aviodance
        self.cbf_symbolic = distance - self.margin
        self.cbf = lambdify([self.robot_state, self.obstacle_state], self.cbf_symbolic)

    def get_relative_position(self, robot_state, obstacle_state):
        """ get the coordinate of obstacle in the robot frame """
        R_inverse = np.array([[cos(robot_state[2]), sin(robot_state[2])], 
                              [-sin(robot_state[2]), cos(robot_state[2])]])
        relative_position = np.array([[obstacle_state[0] - robot_state[0]], 
                                      [obstacle_state[1] - robot_state[1]]])
        # get position in the robot frame
        relative_position = R_inverse @ relative_position
        relative_position = relative_position.reshape(2, )
        return relative_position

    def get_cbf_value(self, relative_position):
        """ get the cbf_value based on the relative position """
        dx = abs(relative_position[0]) - self.rec_width
        dy = abs(relative_position[1]) - self.rec_height

        distance_inside = min(max(dx, dy), 0.0)
        distance_outside = sqrt(max(dx, 0) ** 2 + max(dy, 0) ** 2)
        distance = distance_outside + distance_inside

        # add margin for collision avoidance
        distance = distance - self.margin
        return distance

    def derive_cbf_gradient(self, robot_state, obstacle_state):
        """ 
        derive the gradient of sdf between the rectangle robot and a point from the rectangle osbatcle
        Args:
            robot_state: x, y, theta in world frame
            obstacle_state: ox, oy, ovx, ovy only one point of polytopic-shaped obstacle in the world frame
        Returns:
            lf_cbf in shape ()
            lg_cbf in shape (1, 2)
            dt_obs_cbf in shape ()
        """
        # x_b = R^(-1) * (x_o - xr)
        # dh / dx = [dh / dp, dh / d theta]

        # calculate the dh / dx_b based on numerical values, and use dh / dx_b to express other terms
        dh_dxb = self.get_dh_dxb(robot_state, obstacle_state)

        # lf_cbf, lg_cbf, dt_obs_cbf (dynamic obstacle)
        lf_cbf, lg_cbf = self.get_cbf_gradient(robot_state, obstacle_state, dh_dxb)
        dt_obs_cbf = self.get_dt_obs_cbf(robot_state, obstacle_state, dh_dxb)

        return lf_cbf, lg_cbf, dt_obs_cbf

    def get_dh_dxb(self, robot_state, obstacle_state):
        """ calculate the sdf gradient based on numerical values """

        # method1, based on world frame
        # sdf = self.cbf(robot_state, obstacle_state)
        # sdf_gradient = np.zeros((obstacle_state[0:2].shape[0], ))
        # for i in range(obstacle_state[0:2].shape[0]):
        #     e = np.zeros((obstacle_state.shape[0], ))
        #     e[i] = self.e0
        #     sdf_next = self.cbf(robot_state, obstacle_state + e)
        #     sdf_gradient[i] = (sdf_next - sdf) / self.e0

        # method2, based on robot frame
        relative_position = self.get_relative_position(robot_state, obstacle_state)
        sdf = self.get_cbf_value(relative_position)
        sdf_gradient = np.zeros((obstacle_state[0:2].shape[0], ))
        for i in range(obstacle_state[0:2].shape[0]):
            e = np.zeros((obstacle_state[0:2].shape[0], ))
            e[i] = self.e0
            sdf_next = self.get_cbf_value(relative_position + e)
            sdf_gradient[i] = (sdf_next - sdf) / self.e0
        
        return sdf_gradient

    def get_cbf_gradient(self, robot_state, obstacle_state, sdf_gradient):
        """ 
        get lf_cbf, lg_cbf 
        Args:
            robot_state: x, y, theta
            obstacle_state: ox, oy, ovx, ovy
            sdf_gradient: dh / dx_b
        Returns:
            lf_cbf in shape ()
            lg_cbf in shape (1, 2)
        """
        # h is calculated by the relative position of obstacle
        # dh / dt = (dh / dx) * (dx / dt) x is a vector and dx / dt = f + g * u
        # x_b is in robot frame, x_o is in world frame, xr is the robot state
        # x_b = R^(-1) * (x_o - xr)
        # dh / dx = [dh / dp, dh / d theta] p is xr (x, y)

        # in shape (2, )
        dh_dp = self.get_dh_dp(robot_state, sdf_gradient)
        dh_dtheta = self.get_dh_dtheta(robot_state, obstacle_state, sdf_gradient)
        dh_dx = np.array([dh_dp[0], dh_dp[1], dh_dtheta])

        # get lf_cbf, lg_cbf
        lf_cbf = (dh_dx @ self.f(robot_state))[0]
        lg_cbf = (dh_dx @ self.g(robot_state)).reshape(1, 2)
        return lf_cbf, lg_cbf

    def get_dh_dp(self, robot_state, sdf_gradient):
        """ get dh_dp in shape (2, ) """

        # x_b = R^(-1) * (x_o - xr)
        # dh / d xr = (d x_b / d xr) @ (dh / d x_b) = -R @ (dh / d x_b)
        R = np.array([[cos(robot_state[2]), -sin(robot_state[2])],
                      [sin(robot_state[2]), cos(robot_state[2])]])
        
        dh_dp = -R @ sdf_gradient.reshape(2, 1)
        dh_dp = dh_dp.reshape(2, )

        return dh_dp
    
    def get_dh_dtheta(self, robot_state, obstacle_state, sdf_gradient):
        """ get dh_dtheta in shape () """

        # dh / d theta = (d x_b / d theta)^T @ (dh / d x_b)
        # x_b = R^(-1) * (x_o - xr)
        # d x_b / d theta = d R^-1 / d theta @ (x_o - xr)
        relative_position = np.array([obstacle_state[0] - robot_state[0], obstacle_state[1] - robot_state[1]])
        dR_inverse_dtheta = np.array([[-sin(robot_state[2]), cos(robot_state[2])], 
                                      [-cos(robot_state[2]), -sin(robot_state[2])]])
        dxb_dtheta = dR_inverse_dtheta @ relative_position.reshape(2, 1)

        # in shape (1, 1)
        dh_dtheta = dxb_dtheta.T @ sdf_gradient.reshape(2, 1)
        dh_dtheta = dh_dtheta[0, 0]
        return dh_dtheta

    def get_dt_obs_cbf(self, robot_state, obstacle_state, sdf_gradient):
        """ 
        get the dt_cbf with respect to dynamic obstacle, assume the velocity of obstacle is a constant
        Args:
            robot_state: x, y ,theta
            obstacle_state: ox, oy, ovx, ovy
            sdf_gradient: dh / d x_b
        Returns:
            dt_obs_cbf in shape ()
        """

        # x_b = R^(-1) * (x_o - xr)
        # dh / dt = (dh / d x_o) @ (d x_o / dt)
        # dh / d x_o = (d x_b / d x_o) @ (dh / d x_b)
        # d x_b / d x_o = R
        # dh /dt = (R @ (dh / d x_b)) @ (d x_o / dt)

        R = np.array([[cos(robot_state[2]), -sin(robot_state[2])],
                      [sin(robot_state[2]), cos(robot_state[2])]])
        
        # in shape (2, 1)
        dh_dox = R @ sdf_gradient.reshape(2, 1)
        dt_obs_cbf = dh_dox.reshape(2, ) @ self.obstacle_dynamics(obstacle_state)[0:2]
        
        return dt_obs_cbf[0]
        
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
    # obstacle_vertex = np.array(obs_params['obs_vertexes'])
    # obstacle_vel = np.array(obs_params['obs_vel'])
    for i in np.arange(0, 2, 0.1):
        robot_state = np.array([0.5, 0.6, i])
        # target_state = np.array([3.0, 3.0, 0.2])
        obstacle_state = np.array([2.0, 2.0, 0.2, 0.2])
        a = test_target.get_dh_dxb(robot_state, obstacle_state)
        # print(a)

    # sampled_points = test_target.get_sampled_points_from_obstacle_vertexes(obstacle_vertex, 6)
    # for i in range(sampled_points.shape[0]):
    #     current_point_state = np.array([sampled_points[i][0], sampled_points[i][1], obstacle_vel[0], obstacle_vel[1]])
    #     print(current_point_state)