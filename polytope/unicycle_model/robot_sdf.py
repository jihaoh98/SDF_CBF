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
        circle_obstacle_state: ox, oy, ovx, ovy, o_radius
        polytope_obstacle_state: ox, oy, ovx, ovy
        define the clf and cbf based on sdf for rectangle-shaped robot with respect to any-shaped obstacle
        """
        # robot system states, half width with half height
        self.state_dim = 3
        self.control_dim = 2
        self.rec_width = params['width']
        self.rec_height = params['height']
        self.margin = params['margin']
        self.e0 = float(params['e0'])

        # robot's current state and target state
        x, y, theta = sp.symbols('x y theta')
        self.robot_state = sp.Matrix([x, y, theta])
        e_x, e_y, e_theta = sp.symbols('e_x e_y, e_theta')
        self.target_state = sp.Matrix([e_x, e_y, e_theta])

        # obstacle state
        o_x, o_y, o_vx, o_vy, o_radius = sp.symbols('o_x o_y o_vx o_vy o_radius')

        # circular-shaped obstacle
        self.cir_obs_dim = 5
        self.cir_obstacle_state = sp.Matrix([o_x, o_y, o_vx, o_vy, o_radius])

        # other shaped obstacle (in fact is the point)
        self.obs_dim = 4
        self.obstacle_state = sp.Matrix([o_x, o_y, o_vx, o_vy])

        # robot system dynamics
        self.f = None
        self.f_symbolic = None
        self.g = None
        self.g_symbolic = None

        # obstacle dynamics, single integral system
        self.obstacle_dynamics = None
        self.obstacle_dynamics_symbolic = None
        self.cir_obstacle_dynamics = None
        self.cir_obstacle_dynamics_symbolic = None

        # clf design
        # consider two clfs (rear axle axis can also handle with one clf)
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

        # cbf design, distance of a point to a rectangle
        # lf_cbf and lg_cbf get through numerical results
        self.cbf = None
        self.cbf_symbolic = None
        self.cir_cbf = None
        self.cir_cbf_symbolic = None

        # initialize
        self.init_system()

    def init_system(self):
        """ init the system's dynamics and clf & cbf """
        # init the robot system dynamics and obstacle dynamics
        self.f_symbolic, self.g_symbolic = self.define_system_dynamics()
        self.f = lambdify([self.robot_state], self.f_symbolic)
        self.g = lambdify([self.robot_state], self.g_symbolic)

        # constant velocity and add radius to the circular-shaped obstacle's dynamics
        self.obstacle_dynamics_symbolic = sp.Matrix([self.obstacle_state[2], self.obstacle_state[3], 0.0, 0.0])
        self.obstacle_dynamics = lambdify([self.obstacle_state], self.obstacle_dynamics_symbolic)

        self.cir_obstacle_dynamics_symbolic = sp.Matrix([self.cir_obstacle_state[2], self.cir_obstacle_state[3], 0.0, 0.0, 0.0])
        self.cir_obstacle_dynamics = lambdify([self.cir_obstacle_state], self.cir_obstacle_dynamics_symbolic)

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
        """ init the control lyapunov functions for distance and orientation """
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

    def init_cbf(self):
        """
        init the cbf based on sdf, thus to calculate the sdf between the rectangle-shaped robot and the obstacle
        Args:
        rectangle robot params: center_pose (x, y, theta), half width and height
        obs params: state of a point in the edge of the obstacle (ox, oy, ovx, ovy) (in world frame)
        circle obs params: state of the circular obstacle (ox, oy, ovx, ovy, o_radius) (in the world frame)
        Returns:
            sdf between rectangle and obstacle: float
        """
        # get obs's position in the robot frame
        R_inverse = sp.Matrix([[sp.cos(self.robot_state[2]), sp.sin(self.robot_state[2])],
                               [-sp.sin(self.robot_state[2]), sp.cos(self.robot_state[2])]])
        
        relative_position = sp.Matrix([[self.obstacle_state[0] - self.robot_state[0]],
                                       [self.obstacle_state[1] - self.robot_state[1]]])
        relative_position = R_inverse @ relative_position

        dx = sp.Abs(relative_position[0]) - self.rec_width
        dy = sp.Abs(relative_position[1]) - self.rec_height

        distance_inside = sp.Min(sp.Max(dx, dy), 0.0)
        distance_outside = sp.sqrt(sp.Max(dx, 0) ** 2 + sp.Max(dy, 0) ** 2)
        distance = distance_outside + distance_inside

        # for point of obstacle, without minus obstacle_radius, add margin for collision aviodance
        self.cbf_symbolic = distance - self.margin
        self.cbf = lambdify([self.robot_state, self.obstacle_state], self.cbf_symbolic)

        # for circlular-shaped obstacle, minus radius
        self.cir_cbf_symbolic = distance - self.cir_obstacle_state[4] - self.margin
        self.cir_cbf = lambdify([self.robot_state, self.cir_obstacle_state], self.cir_cbf_symbolic)

    def get_relative_position(self, robot_state, obstacle_state):
        """ get the coordinate of obstacle in the robot frame """
        R_inverse = np.array([[cos(robot_state[2]), sin(robot_state[2])], 
                              [-sin(robot_state[2]), cos(robot_state[2])]])
        relative_position = np.array([[obstacle_state[0] - robot_state[0]], 
                                      [obstacle_state[1] - robot_state[1]]])
        # get position in the robot frame
        relative_position = R_inverse @ relative_position
        relative_position = relative_position.reshape(2, )

        # in shape (2, )
        return relative_position

    def get_cbf_value(self, relative_position, obs_shape=None, obstacle_radius=None):
        """ get the cbf_value based on the relative position """
        dx = abs(relative_position[0]) - self.rec_width
        dy = abs(relative_position[1]) - self.rec_height

        distance_inside = min(max(dx, dy), 0.0)
        distance_outside = sqrt(max(dx, 0) ** 2 + max(dy, 0) ** 2)
        distance = distance_outside + distance_inside

        # add margin for collision avoidance
        if obs_shape == 'circle':
            distance = distance - obstacle_radius - self.margin
        else:
            distance = distance - self.margin
        return distance
    
    def derive_cbf_gradient_direct(self, robot_state, obstacle_state, obs_shape=None):
        """ get the cbf gradient directly """
        if obs_shape == 'circle':
            sdf = self.cir_cbf(robot_state, obstacle_state)
        else:
            sdf = self.cbf(robot_state, obstacle_state)

        # robot cbf gradient
        sdf_gradient = np.zeros((robot_state.shape[0], ))
        for i in range(robot_state.shape[0]):
            e = np.zeros((robot_state.shape[0], ))
            e[i] = self.e0
            if obs_shape == 'circle':
                sdf_next = self.cir_cbf(robot_state + e, obstacle_state)
            else:
                sdf_next = self.cbf(robot_state + e, obstacle_state)
            sdf_gradient[i] = (sdf_next - sdf) / self.e0
        
        lf_cbf = (sdf_gradient @ self.f(robot_state))[0]
        lg_cbf = (sdf_gradient @ self.g(robot_state)).reshape(1, 2)

        # for dynamic obstacle
        obs_gradient = np.zeros((2, ))
        for i in range(2):
            e = np.zeros((obstacle_state.shape[0], ))
            e[i] = self.e0
            if obs_shape == 'circle':
                sdf_next = self.cir_cbf(robot_state, obstacle_state + e)
            else:
                sdf_next = self.cbf(robot_state, obstacle_state + e)
            obs_gradient[i] = (sdf_next - sdf) / self.e0

        # no difference for different obstacles
        dt_obs_cbf = obs_gradient @ self.obstacle_dynamics(obstacle_state[0:4])[0:2]
        return lf_cbf, lg_cbf, dt_obs_cbf[0]

    def derive_cbf_gradient(self, robot_state, obstacle_state, obs_shape=None):
        """ 
        derive the gradient of sdf between the rectangle robot and a point from the polytopic-shaped obstacle (or circular obstacle)
        Args:
            robot_state: x, y, theta in world frame
            obstacle_state: [ox, oy, ovx, ovy], only one point of polytopic-shaped obstacle in the world frame
            cir_obstacle_state: [ox, oy, ovx, ovy, obstacle radius] for circular obstacle
        Returns:
            lf_cbf in shape ()
            lg_cbf in shape (1, 2)
            dt_obs_cbf in shape ()
        """
        # x_b = R^(-1) * (x_o - xr)
        # dh / dx = [dh / dp, dh / dtheta]

        # calculate the dh / dx_b based on numerical values, and use dh / dx_b to express other terms
        dh_dxb = self.get_dh_dxb_relative(robot_state, obstacle_state, obs_shape)

        # lf_cbf, lg_cbf, dt_obs_cbf (dynamic obstacle)
        lf_cbf, lg_cbf = self.get_cbf_gradient(robot_state, obstacle_state, dh_dxb)
        dt_obs_cbf = self.get_dt_obs_cbf(robot_state, obstacle_state, dh_dxb)

        return lf_cbf, lg_cbf, dt_obs_cbf
    
    def get_dh_dxb(self, robot_state, obstacle_state, obs_shape=None):
        """ calculate the sdf gradient based on numerical values """
        # method1, based on world frame, not correct
        if obs_shape == 'circle':
            sdf = self.cir_cbf(robot_state, obstacle_state)
        else:
            sdf = self.cbf(robot_state, obstacle_state)

        sdf_gradient = np.zeros((obstacle_state[0:2].shape[0], ))
        for i in range(obstacle_state[0:2].shape[0]):
            e = np.zeros((obstacle_state.shape[0], ))
            e[i] = self.e0
            if obs_shape == 'circle':
                sdf_next = self.cir_cbf(robot_state, obstacle_state + e)
            else:
                sdf_next = self.cbf(robot_state, obstacle_state + e)
            sdf_gradient[i] = (sdf_next - sdf) / self.e0

        return sdf_gradient
    
    def get_dh_dxb_relative(self, robot_state, obstacle_state, obs_shape=None):
        """ calculate the sdf gradient based on numerical values """
        # method2, based on robot frame
        relative_position = self.get_relative_position(robot_state, obstacle_state)
        if obs_shape == 'circle':
            sdf = self.get_cbf_value(relative_position, obs_shape, obstacle_state[4])
        else:
            sdf = self.get_cbf_value(relative_position)

        sdf_gradient = np.zeros((obstacle_state[0:2].shape[0], ))
        for i in range(obstacle_state[0:2].shape[0]):
            e = np.zeros((obstacle_state[0:2].shape[0], ))
            e[i] = self.e0
            if obs_shape == 'circle':
                sdf_next = self.get_cbf_value(relative_position + e, obs_shape, obstacle_state[4])
            else:
                sdf_next = self.get_cbf_value(relative_position + e)
            sdf_gradient[i] = (sdf_next - sdf) / self.e0
        sdf_gradient[1] = sdf_gradient[1] * 2.0
        return sdf_gradient

    def get_cbf_gradient(self, robot_state, obstacle_state, sdf_gradient):
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
        # x_b = R^(-1) * (x_o - xr)
        # dh / dx = [dh / dp, dh / dtheta] p is xr (x, y)

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
        # dh / dxr = (dx_b / dxr) @ (dh / dx_b) = -R @ (dh / d x_b)
        R = np.array([[cos(robot_state[2]), -sin(robot_state[2])],
                      [sin(robot_state[2]), cos(robot_state[2])]])
        
        dh_dp = -R @ sdf_gradient.reshape(2, 1)
        dh_dp = dh_dp.reshape(2, )
        return dh_dp
    
    def get_dh_dtheta(self, robot_state, obstacle_state, sdf_gradient):
        """ get dh_dtheta in shape () """
        # x_b = R^(-1) * (x_o - xr)
        # dh / dtheta = (dx_b / dtheta)^T @ (dh / dx_b)
        # dx_b / dtheta = d R^-1 / d theta @ (x_o - xr)

        relative_position = np.array([obstacle_state[0] - robot_state[0], 
                                      obstacle_state[1] - robot_state[1]])
        dR_dtheta = np.array([[-sin(robot_state[2]), -cos(robot_state[2])], 
                               [cos(robot_state[2]), -sin(robot_state[2])]])
        
        dxb_dtheta = relative_position.reshape(1, 2) @ dR_dtheta
        
        # in shape (1, 1)
        dh_dtheta = dxb_dtheta @ sdf_gradient.reshape(2, 1)
        dh_dtheta = dh_dtheta[0, 0]
        return dh_dtheta

    def get_dt_obs_cbf(self, robot_state, obstacle_state, sdf_gradient):
        """ 
        get the dt_cbf with respect to the dynamic obstacle, assume the velocity of obstacle is a constant
        Args:
            robot_state: x, y ,theta
            obstacle_state: ox, oy, ovx, ovy
            cir_obstacle_state: ox, oy, ovx, ovy, o_radius
            sdf_gradient: dh / dx_b
        Returns:
            dt_obs_cbf in shape ()
        """
        # x_b = R^(-1) * (x_o - xr)
        # dh / dt = (dh / dx_o) @ (dx_o / dt)
        # dh / dx_o = (dx_b / dx_o) @ (dh / dx_b)
        # dx_b / d x_o = R
        # dh /dt = (R @ (dh / d x_b)) @ (d x_o / dt)

        R = np.array([[cos(robot_state[2]), -sin(robot_state[2])],
                      [sin(robot_state[2]), cos(robot_state[2])]])
        
        # in shape (2, 1)
        dh_dox = R @ sdf_gradient.reshape(2, 1)
        # no different for different obstacle
        dt_obs_cbf = dh_dox.reshape(2, ) @ self.obstacle_dynamics(obstacle_state[0:4])[0:2] 
        
        # dt_obs_cbf in shape (1, )
        return dt_obs_cbf[0]
    
    def get_sampled_points_from_obstacle_vertexes(self, obstacle_vertexes, num_samples):
        """
        get the sample points from the obstacle vertexes
        Params:
            obstacle vextexes: obstacle vertexes position in anticlockwise np.array (n, 2)
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
    # cir_obs_params = config['cir_obstacle_list']
    # obs_params = config['obstacle_list']
    test_target = Robot_Sdf(robot_params)

    # for i in np.arange(0, 2, 0.1):
    #     robot_state = np.array([0.5, 0.6, i])
    #     # target_state = np.array([3.0, 3.0, 0.2])
    #     obstacle_state = np.array([2.0, 2.0, 0.2, 0.2, 0.5])
    #     a = test_target.get_dh_dxb(robot_state, obstacle_state)
    #     print(a)

    robot_state = np.array([0.5, 0.6, 0.5])
    obstacle_state = np.array([2.5, 2.1, 0.2, 0.2])
    cir_obstacle_state = np.array([2.5, 2.1, 0.2, 0.2, 0.5])
    # print(test_target.cbf(robot_state, obstacle_state))

    print(test_target.cbf(robot_state, obstacle_state))
    print(test_target.derive_cbf_gradient(robot_state, obstacle_state))
    print(test_target.derive_cbf_gradient_direct(robot_state, obstacle_state))

    print(test_target.cir_cbf(robot_state, cir_obstacle_state))
    print(test_target.derive_cbf_gradient(robot_state, cir_obstacle_state, 'circle'))
    print(test_target.derive_cbf_gradient_direct(robot_state, cir_obstacle_state, 'circle'))


    # print(test_target.next_state(robot_state, [0.1, 0.1], 0.1))