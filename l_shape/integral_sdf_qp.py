import numpy as np
import casadi as ca
from integral_robot_sdf import Integral_Robot_Sdf
import yaml


class Integral_Sdf_Cbf_Clf:
    def __init__(self, file_name) -> None:
        """ init the optimal problem with clf-cbf-qp """
        with open(file_name) as file:
            config = yaml.safe_load(file)
        
        controller_params = config['controller']
        robot_params = config['robot']

        # init robot
        self.robot = Integral_Robot_Sdf(robot_params)
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
        self.cir_cbf = self.robot.cir_cbf

        # get the parameter for optimal control
        self.weight_input = controller_params['weight_input']
        self.smooth_input = controller_params['smooth_input']
        self.weight_slack = controller_params['weight_slack']
        self.clf_lambda = controller_params['clf_lambda']
        self.cbf_gamma = controller_params['cbf_gamma']

        # boundary of control variables
        self.u_max = robot_params['u_max']
        self.u_min = robot_params['u_min']

        # TODO
        # for optimal-decay cbf
        # self.w0 = controller_params['w0']
        # self.weight_w = controller_params['weight_w']
    
        # optimize and set the solver

        # ipopt
        # self.opti = ca.Opti()
        # opts_setting = {
        #         'ipopt.max_iter': 100,
        #         'ipopt.print_level': 0,
        #         'print_time': 0,
        #         'ipopt.acceptable_tol': 1e-8,
        #         'ipopt.acceptable_obj_change_tol': 1e-6
        #     }
        # self.opti.solver('ipopt', opts_setting)

        # qpoases
        self.opti = ca.Opti('conic')
        opts_setting = {
            'printLevel': 'low',  
            'error_on_fail': False,
            'expand': True,
            'print_time': 0
        }
        self.opti.solver('qpoases', opts_setting)

        self.u = self.opti.variable(self.control_dim)
        self.slack = self.opti.variable()
        self.obj = None
  
        # approach the desired control and smooth the control
        self.H = np.diag(self.weight_input)
        self.R = np.diag(self.smooth_input)

    def set_optimal_function(self, u_ref, add_slack=True):
        """ set the optimal function """
        self.obj = (self.u - u_ref).T @ self.H @ (self.u - u_ref)
        if add_slack:
            self.obj = self.obj + self.weight_slack * self.slack ** 2
        self.opti.minimize(self.obj)
        self.opti.subject_to()

    def add_clf_cons(self, robot_cur_state, add_slack=True):
        """ add clf cons """
        clf = self.clf(robot_cur_state, self.target_state)
        lf_clf = self.lf_clf(robot_cur_state, self.target_state)
        lg_clf = self.lg_clf(robot_cur_state, self.target_state)

        if add_slack:
            self.opti.subject_to(lf_clf + (lg_clf @ self.u)[0, 0] + self.clf_lambda * clf <= self.slack)
            self.opti.subject_to(self.opti.bounded(-np.inf, self.slack, np.inf))
        else:
            self.opti.subject_to(lf_clf + (lg_clf @ self.u)[0, 0] + self.clf_lambda * clf <= 0)
        
        return clf

    def add_controls_physical_cons(self):
        """ add physical constraint of controls """
        self.opti.subject_to(self.opti.bounded(self.u_min, self.u, self.u_max))

    def add_cir_cbf_cons(self, robot_cur_state, robot_params, two_center, cir_obs_state):
        """ add cons w.r.t circle obstacle """

        cbf = self.cir_cbf(robot_cur_state, robot_params, two_center, cir_obs_state)
        lf_cbf, lg_cbf, dt_obs_cbf = self.robot.derive_cbf_gradient(robot_cur_state, robot_params, two_center, cir_obs_state, obs_shape='circle')
        self.opti.subject_to(lf_cbf + (lg_cbf @ self.u)[0, 0] + dt_obs_cbf + self.cbf_gamma * cbf >= 0)

        return cbf
                
    def add_cbf_cons(self, robot_cur_state, robot_params, two_center, obs_state, obs_vertex):
        """ add cons w.r.t other-shaped obstacle """
        min_sdf = 10000 
        # sample points from obstacles and add constraints to the optimal problem
        sampled_points = self.robot.get_sampled_points_from_obstacle_vertexes(obs_vertex, num_samples=6)
        
        # TODO, reduce the number of points
        for i in range(sampled_points.shape[0]):    
            obs_cur_point_state = np.array([sampled_points[i][0], sampled_points[i][1], obs_state[2], obs_state[3]])
            cbf = self.cbf(robot_cur_state, robot_params, two_center, obs_cur_point_state)
            lf_cbf, lg_cbf, dt_obs_cbf = self.robot.derive_cbf_gradient(robot_cur_state, robot_params, two_center, obs_cur_point_state)
            self.opti.subject_to(lf_cbf + (lg_cbf @ self.u)[0, 0] + dt_obs_cbf + self.cbf_gamma * cbf >= 0)

            if cbf < min_sdf:
                min_sdf = cbf

        return min_sdf
    
    def clf_qp(self, robot_cur_state, add_slack=False, u_ref=None):
        """ 
        calculate the optimal control which navigating the robot to its destination
        Args: 
            robot_cur_state: [x, y, theta] np.array(3, )
            u_ref: None or np.array(2, )
        Returns:
            optimal control u
        """
        if u_ref is None:
            u_ref = np.zeros(self.control_dim)

        self.set_optimal_function(u_ref, add_slack)        
        clf = self.add_clf_cons(robot_cur_state, add_slack)
        self.add_controls_physical_cons()

        # optimize the qp problem
        try:
            sol = self.opti.solve()
            optimal_control = sol.value(self.u)
            return optimal_control, clf, True
        except:
            print(self.opti.return_status() + ' clf qp')
            return None, None, False
        
    def cbf_clf_qp(self, robot_cur_state, robot_params, two_center, obs_states=None, obs_vertexes=None, cir_obs_states=None, add_clf=True, u_ref=None):
        """
        This is a function to calculate the optimal control for the robot with respect to different shaped obstacles
        Args:
            robot_cur_state: [x, y, theta] np.array(3, )
            robot_params: [width, height] np.array(2, )
            two_center: [x1, y1, x2, y2] np.array(4, )
            obs_state: [ox, oy, ovx, ovy] list for obstacle state
            cir_obs_state: [ox, oy, ovx, ovy, o_radius] list for circle obstacle state
        Returns:
            optimal control u
        """
        if u_ref is None:
            u_ref = np.zeros(self.control_dim)
        self.set_optimal_function(u_ref, add_clf)

        clf = None
        if add_clf:
            clf = self.add_clf_cons(robot_cur_state, add_slack=True)

        # for plot
        cbf_list = None
        cir_cbf_list = None

        # add cbf constraints for each obstacles
        # circular-shaped obstacles
        if cir_obs_states is not None:
            cir_cbf_list = []
            for cir_obs_state in cir_obs_states:
                cbf = self.add_cir_cbf_cons(robot_cur_state, robot_params, two_center, cir_obs_state)
                cir_cbf_list.append(cbf)
                
        # other-shaped obstacles
        if obs_states is not None:
            cbf_list = []
            obs_num = len(obs_states)
            for i in range(obs_num):
                cbf = self.add_cbf_cons(robot_cur_state, robot_params, two_center, obs_states[i], obs_vertexes[i])
                cbf_list.append(cbf)
    
        self.add_controls_physical_cons()
        
        # optimize the qp problem
        try:
            sol = self.opti.solve()
            optimal_control = sol.value(self.u)
            if add_clf:
                slack = sol.value(self.slack)
                return optimal_control, cbf_list, cir_cbf_list, clf, slack, True
            else:
                return optimal_control, cbf_list, cir_cbf_list, None, None, True
        except:
            print(self.opti.return_status() + ' sdf-cbf with clf')
            return None, cbf_list, cir_cbf_list, None, None, False
