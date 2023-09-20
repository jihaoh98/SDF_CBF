import numpy as np
import casadi as ca
import robot_sdf
import yaml


class Sdf_Cbf:
    def __init__(self) -> None:
        """ init the optimal problem """
        file_name = 'settings.yaml'
        with open(file_name) as file:
            config = yaml.safe_load(file)
        
        controller_params = config['controller']
        robot_params = config['robot']

        # init robot
        self.robot = robot_sdf.Robot_Sdf(robot_params)
        self.state_dim = self.robot.state_dim 
        self.control_dim = self.robot.control_dim
        self.target_state = robot_params['target_state']
        self.sensor_range = robot_params['sensor_range']

        # initialize CLF
        self.clf = self.robot.clf
        self.lf_clf = self.robot.lf_clf
        self.lg_clf = self.robot.lg_clf

        # initialize CBF
        self.cbf = self.robot.cbf

        # get the parameter for optimal control
        self.weight_input = controller_params['weight_input']
        self.smooth_input = controller_params['smooth_input']
        self.weight_slack = controller_params['weight_slack']
        self.clf_lambda = controller_params['clf_lambda']
        self.cbf_gamma = controller_params['cbf_gamma']

        # boundary of control variables
        self.u_max = robot_params['u_max']
        self.u_min = robot_params['u_min']

        # for optimal-decay cbf
        # self.w0 = controller_params['w0']
        # self.weight_w = controller_params['weight_w']
    
        # optimize and set the solver
        self.opti = ca.Opti()
        opts_setting = {
                'ipopt.max_iter': 100,
                'ipopt.print_level': 0,
                'print_time': 0,
                'ipopt.acceptable_tol': 1e-8,
                'ipopt.acceptable_obj_change_tol': 1e-6
            }
        self.opti.solver('ipopt', opts_setting)

        self.u = self.opti.variable(self.control_dim)
        self.slack = self.opti.variable()
        self.obj = None
  
        # approach the desired control and smooth the control
        self.H = np.diag(self.weight_input)
        self.R = np.diag(self.smooth_input)
        
    def cbf_clf_qp(self, robot_cur_state, obs_veretxes, obs_state, add_clf=True, u_ref=None):
        """
        This is a function to calculate the optimal control for the robot with respect to the polytopic-shaped obstacle
        Args:
            robot_cur_state: [x, y] np.array(2, )
            obs: class of obstacles, to get state and vertexes, many obstacles
        Returns:
            optimal control u
        """
        if u_ref is None:
            u_ref = np.zeros(self.control_dim)

        self.obj = (self.u - u_ref).T @ self.H @ (self.u - u_ref)
        if add_clf:
            self.obj = self.obj + self.weight_slack * self.slack ** 2
        self.opti.minimize(self.obj)
        self.opti.subject_to()

        # clf constraint
        if add_clf:
            clf = self.clf(robot_cur_state, self.target_state)
            lf_clf = self.lf_clf(robot_cur_state, self.target_state)
            lg_clf = self.lg_clf(robot_cur_state, self.target_state)

            self.opti.subject_to(lf_clf + (lg_clf @ self.u)[0, 0] + self.clf_lambda * clf <= self.slack)
            self.opti.subject_to(self.opti.bounded(-np.inf, self.slack, np.inf))

        # cbf constraint
        # sample points from obstacles and add constraints to the optimal problem
        obs_num = len(obs_veretxes)
        for i in range(obs_num):
            sampled_points = self.robot.get_sampled_points_from_obstacle_vertexes(obs_veretxes[i], num_samples=6)

            for j in range(sampled_points.shape[0]):
                current_point_state = np.array([sampled_points[j][0], sampled_points[j][1], obs_state[i][2], obs_state[i][3]])
                # print(current_point_state)
                cbf = self.cbf(robot_cur_state, current_point_state)
                lf_cbf, lg_cbf, dt_obs_cbf = self.robot.derive_cbf_gradient(robot_cur_state, current_point_state)
                self.opti.subject_to(lf_cbf + (lg_cbf @ self.u)[0, 0] + dt_obs_cbf + self.cbf_gamma * cbf >= 0)
        
        # set the boundary constraints
        self.opti.subject_to(self.opti.bounded(self.u_min, self.u, self.u_max))
        
        # optimize the qp problem
        try:
            sol = self.opti.solve()
            optimal_control = sol.value(self.u)
            if add_clf:
                slack = sol.value(self.slack)
                return optimal_control, clf, slack, True
            else:
                return optimal_control, None, None, True
        except:
            print(self.opti.return_status() + ' sdf-cbf with clf')
            return None, None, None, False

