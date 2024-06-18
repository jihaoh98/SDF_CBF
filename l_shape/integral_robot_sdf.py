import numpy as np
import sympy as sp
import yaml
from math import sqrt, cos, sin
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt
from l_shape_robot import L_shaped_robot


class Integral_Robot_Sdf:
    def __init__(self, params) -> None:
        """
        robot state: x, y, theta (only for plot, with constant value zero)
        controls: vx, vy
        circle_obstacle_state: ox, oy, ovx, ovy, o_radius
        polytope_obstacle_state: ox, oy, ovx, ovy
        define the clf and cbf based on sdf for L-shaped robot with respect to any-shaped obstacle
        """
        # robot system states, half width with half height
        self.state_dim = 3
        self.control_dim = 2
        self.margin = params['margin']
        self.e0 = float(params['e0'])

        # robot's current state and target state
        x, y, theta = sp.symbols('x y theta')
        self.robot_state = sp.Matrix([x, y, theta])
        e_x, e_y, e_theta = sp.symbols('e_x e_y, e_theta')
        self.target_state = sp.Matrix([e_x, e_y, e_theta])

        # parameter for L-shaped robot
        # half width and height
        width, height = sp.symbols('width height')
        self.robot_param = sp.Matrix([width, height])

        # two center of L-shape robot
        x1, y1, x2, y2 = sp.symbols('x1, y1, x2, y2')
        self.two_center = sp.Matrix([x1, y1, x2, y2])

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

        # clf and cbf design
        self.clf = None
        self.clf_symbolic = None
        self.lf_clf = None
        self.lf_clf_symbolic = None
        self.lg_clf = None
        self.lg_clf_symbolic = None

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

        self.cir_obstacle_dynamics_symbolic = sp.Matrix(
            [self.cir_obstacle_state[2], self.cir_obstacle_state[3], 0.0, 0.0, 0.0])
        self.cir_obstacle_dynamics = lambdify([self.cir_obstacle_state], self.cir_obstacle_dynamics_symbolic)

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
        init the cbf based on sdf, thus to calculate the sdf between the L-shaped robot and the obstacle
        Args:
        L-shaped robot params: center_pose (x, y, theta), half width and height, two center of L-shaped robot
        obs params: state of a point in the edge of the obstacle (ox, oy, ovx, ovy) (in world frame)
        circle obs params: state of the circular obstacle (ox, oy, ovx, ovy, o_radius) (in the world frame)
        Returns:
            sdf between L-shaped and the obstacle: float
        """
        # get obs's position in the robot frame
        relative_position = sp.Matrix([[self.obstacle_state[0] - self.robot_state[0]],
                                       [self.obstacle_state[1] - self.robot_state[1]]])

        # diatance with the first rectangle (L-shaped robot combined by two rectangles)
        # robot_params: [width and height]
        dx1 = sp.Abs(relative_position[0] - self.two_center[0]) - self.robot_param[0] / 2
        dy1 = sp.Abs(relative_position[1] - self.two_center[1]) - self.robot_param[1] / 2
        distance_inside1 = sp.Min(sp.Max(dx1, dy1), 0.0)
        distance_outside1 = sp.sqrt(sp.Max(dx1, 0) ** 2 + sp.Max(dy1, 0) ** 2)
        distance1 = distance_outside1 + distance_inside1

        # diatance with the second rectangle
        dx2 = sp.Abs(relative_position[0] - self.two_center[2]) - self.robot_param[1] / 2
        dy2 = sp.Abs(relative_position[1] - self.two_center[3]) - self.robot_param[0] / 2
        distance_inside2 = sp.Min(sp.Max(dx2, dy2), 0.0)
        distance_outside2 = sp.sqrt(sp.Max(dx2, 0) ** 2 + sp.Max(dy2, 0) ** 2)
        distance2 = distance_outside2 + distance_inside2

        # both distance small than zero is not considered (have some defects, however, not important)
        distance = sp.Min(distance1, distance2)

        # for point of obstacle, without minus obstacle_radius, add margin for collision aviodance
        self.cbf_symbolic = distance - self.margin
        self.cbf = lambdify([self.robot_state, self.robot_param, self.two_center, self.obstacle_state],
                            self.cbf_symbolic)

        # for circlular-shaped obstacle, minus radius
        self.cir_cbf_symbolic = distance - self.cir_obstacle_state[4] - self.margin
        self.cir_cbf = lambdify([self.robot_state, self.robot_param, self.two_center, self.cir_obstacle_state],
                                self.cir_cbf_symbolic)

    def derive_cbf_gradient(self, robot_state, robot_params, two_center, obstacle_state, obs_shape=None):
        """ 
        derive the gradient of sdf between the L-shaped robot and a point from the polytopic-shaped obstacle (or circular obstacle)
        Args:
            robot_state: [x, y, theta] in the world frame
            robot_parmas: [width, height]
            two_center: [x1, y1, x2, y2]
            obstacle_state: [ox, oy, ovx, ovy], only one point of polytopic-shaped obstacle in the world frame
            cir_obstacle_state: [ox, oy, ovx, ovy, obstacle radius] for circular obstacle

        Returns:
            lf_cbf in shape ()
            lg_cbf in shape (1, 2)
            dt_obs_cbf in shape ()
        """
        # x_b = (x_o - xr)
        # here don't consider orientation
        # dh / dx = [dh / dp, dh / dtheta = 0] 

        # calculate the dh / dx_b based on numerical values, and use dh / dx_b to express other terms
        dh_dxb = self.get_dh_dxb(robot_state, robot_params, two_center, obstacle_state, obs_shape)

        # lf_cbf, lg_cbf, dt_obs_cbf (dynamic obstacle)
        lf_cbf, lg_cbf, dt_obs_cbf = self.get_cbf_gradient(robot_state, obstacle_state, dh_dxb)
        return lf_cbf, lg_cbf, dt_obs_cbf

    def get_dh_dxb(self, robot_state, robot_params, two_center, obstacle_state, obs_shape=None):
        """ calculate the sdf gradient based on numerical values """
        if obs_shape == 'circle':
            sdf = self.cir_cbf(robot_state, robot_params, two_center, obstacle_state)
        else:
            sdf = self.cbf(robot_state, robot_params, two_center, obstacle_state)

        sdf_gradient = np.zeros((obstacle_state[0:2].shape[0],))
        for i in range(obstacle_state[0:2].shape[0]):
            e = np.zeros((obstacle_state.shape[0],))
            e[i] = self.e0
            if obs_shape == 'circle':
                sdf_next = self.cir_cbf(robot_state, robot_params, two_center, obstacle_state + e)
            else:
                sdf_next = self.cbf(robot_state, robot_params, two_center, obstacle_state + e)
            sdf_gradient[i] = (sdf_next - sdf) / self.e0
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
        # x_b = (x_o - xr)
        # dh / dx = [dh / dp, dh / dtheta = 0] p is xr (x, y)

        dh_dp = -sdf_gradient
        dh_dx = np.array([dh_dp[0], dh_dp[1], 0])
        lf_cbf = (dh_dx @ self.f(robot_state))[0]
        lg_cbf = (dh_dx @ self.g(robot_state)).reshape(1, 2)
        dt_obs_cbf = (sdf_gradient @ self.obstacle_dynamics(obstacle_state[0:4])[0:2])[0]

        return lf_cbf, lg_cbf, dt_obs_cbf

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
        next_state = next_state + dt * (
                    self.f(current_state).T[0] + (self.g(current_state) @ np.array(u).reshape(self.control_dim, -1)).T[
                0])

        return next_state


if __name__ == '__main__':
    file_name = 'settings.yaml'
    with open(file_name) as file:
        config = yaml.safe_load(file)

    robot_params = config['robot']
    test_target = Integral_Robot_Sdf(robot_params)
    robot_vertexes = [[1.0, 1.0], [2.0, 1.0], [2.0, 1.5], [1.5, 1.5], [1.5, 2.0], [1.0, 2.0]]
    test_robot = L_shaped_robot(0, robot_vertexes)

    gradient = []
    for i in np.arange(1.98, 2.1, 0.001):
        obstacle_state = np.array([i, 1.7, 0.0, 0.0])

        _, b, _ = test_target.derive_cbf_gradient(test_robot.cur_state, [test_robot.width, test_robot.height],
                                                  test_robot.cur_center_body_frame.reshape(-1, ), obstacle_state)
        gradient.append([b[0, 0], b[0, 1]])
    gradient = np.array(gradient)
    t = np.arange(1.98, 2.1, 0.001)
    plt.plot(t, gradient[:, 0], color='k')
    plt.plot(t, gradient[:, 1], color='b')

    plt.grid()
    plt.show()

    # for i in np.arange(0, 2, 0.1):
    #     robot_state = np.array([0.5, 0.6, i])
    #     # target_state = np.array([3.0, 3.0, 0.2])
    #     obstacle_state = np.array([2.0, 2.0, 0.2, 0.2, 0.5])
    #     a = test_target.get_dh_dxb(robot_state, obstacle_state)
    #     print(a)

    # target_state = np.array([1.0, 2.0, 0.0])
    # obstacle_state = np.array([2.5, 2.1, 0.2, 0.2])
    # cir_obstacle_state = np.array([2.5, 2.1, 0.2, 0.2, 0.5])
