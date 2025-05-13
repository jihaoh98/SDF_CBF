import numpy as np
import casadi as ca
import yaml
from scipy.spatial import ConvexHull
from unicycle_robot_sdf import Unicycle_Robot_Sdf


class Unicycle_Sdf_Cbf_Clf:
    def __init__(self, file_name) -> None:
        """ init the optimal problem with clf-cbf-qp """
        with open(file_name) as file:
            config = yaml.safe_load(file)
        
        controller_params = config['controller']
        robot_params = config['robot']

        # init robot
        self.robot = Unicycle_Robot_Sdf(robot_params)
        self.state_dim = self.robot.state_dim 
        self.control_dim = self.robot.control_dim
        self.target_state = robot_params['target_state']
        self.sensor_range = robot_params['sensor_range']

        # initialize CLFs
        # for distance
        self.clf1 = self.robot.clf1
        self.lf_clf1 = self.robot.lf_clf1
        self.lg_clf1 = self.robot.lg_clf1
        
        # for orientation
        self.clf2 = self.robot.clf2
        self.lf_clf2 = self.robot.lf_clf2
        self.lg_clf2 = self.robot.lg_clf2
        
        # for both
        self.clf3 = self.robot.clf3
        self.lf_clf3 = self.robot.lf_clf3
        self.lg_clf3 = self.robot.lg_clf3
        
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
        self.opti = ca.Opti()
        opts_setting = {
                'ipopt.max_iter': 5000,
                'ipopt.print_level': 0,
                'print_time': 0,
                'ipopt.acceptable_tol': 1e-8,
                'ipopt.acceptable_obj_change_tol': 1e-6
            }
        self.opti.solver('ipopt', opts_setting)

        # qpoases
        # self.opti = ca.Opti('conic')
        # opts_setting = {
        #     'printLevel': 'low',  
        #     'error_on_fail': False,
        #     'expand': True,
        #     'print_time': 0
        # }
        # self.opti.solver('qpoases', opts_setting)

        self.u = self.opti.variable(self.control_dim)
        self.lam_A_dot = self.opti.variable(1, 4)
        self.lam_B_dot = self.opti.variable(1, 4)
        self.lam_G_dot = self.opti.variable(1, 4)
        self.slack = self.opti.variable()
        self.obj = None
  
        # approach the desired control and smooth the control
        self.H = np.diag(self.weight_input)
        self.R = np.diag(self.smooth_input)

    def set_optimal_function(self, u_ref, add_slack=True):
        """ set the optimal function """
        self.obj = (self.u - u_ref).T @ self.H @ (self.u - u_ref)
        if add_slack:
            # self.obj = self.obj
            self.obj = self.obj + self.weight_slack * self.slack ** 2
        self.opti.minimize(self.obj)

        self.opti.subject_to(self.u[0] >= -0.3)
        self.opti.subject_to(self.u[0] <= 0.3)
        self.opti.subject_to(self.u[1] >= -0.2)
        self.opti.subject_to(self.u[1] <= 0.2)

    # def add_clf_distance_cons(self, robot_cur_state, add_slack=True):
    #     """ add clf cons of distance """
    #     clf1 = self.clf1(robot_cur_state, self.target_state)
    #     lf_clf1 = self.lf_clf1(robot_cur_state, self.target_state)
    #     lg_clf1 = self.lg_clf1(robot_cur_state, self.target_state)

    #     if add_slack:
    #         self.opti.subject_to(lf_clf1 + (lg_clf1 @ self.u)[0, 0] + self.clf_lambda[0] * clf1 <= self.slack1)
    #         self.opti.subject_to(self.opti.bounded(-np.inf, self.slack1, np.inf))
    #     else:
    #         self.opti.subject_to(lf_clf1 + (lg_clf1 @ self.u)[0, 0] + self.clf_lambda[0] * clf1 <= 0)
        
    #     return clf1
    
    # def add_clf_theta_cons(self, robot_cur_state, add_slack=True):
    #     """ add clf cons of theta """
    #     clf2 = self.clf2(robot_cur_state, self.target_state)
    #     lf_clf2 = self.lf_clf2(robot_cur_state, self.target_state)
    #     lg_clf2 = self.robot.lg_clf2(robot_cur_state, self.target_state)

    #     if lg_clf2[0, 1] != 0:
    #         if add_slack:
    #             self.opti.subject_to(lf_clf2 + (lg_clf2 @ self.u)[0, 0] + self.clf_lambda[1] * clf2 <= self.slack2)
    #             self.opti.subject_to(self.opti.bounded(-np.inf, self.slack2, np.inf))
    #         else:
    #             self.opti.subject_to(lf_clf2 + (lg_clf2 @ self.u)[0, 0] + self.clf_lambda[1] * clf2 <= 0)

    #     return clf2
    
    def add_clf_dist_theta_cons(self, robot_cur_state, add_slack=False):
        """ add clf cons of theta """
        clf3 = self.clf3(robot_cur_state, self.target_state)
        lf_clf3 = self.lf_clf3(robot_cur_state, self.target_state)
        lg_clf3 = self.robot.lg_clf3(robot_cur_state, self.target_state)

        if lg_clf3[0, 1] != 0:
            if add_slack:
                self.opti.subject_to(lf_clf3 + (lg_clf3 @ self.u)[0, 0] + self.clf_lambda[1] * clf3 <= self.slack)
                # self.opti.subject_to(self.opti.bounded(-np.inf, self.slack2, np.inf))
            else:
                self.opti.subject_to(lf_clf3 + (lg_clf3 @ self.u)[0, 0] + self.clf_lambda[1] * clf3 <= 0)

        # return clf3

        # term_1 = (robot_cur_state[:2] - self.target_state[:2]) @ np.array([np.cos(robot_cur_state[2] + np.pi/4), np.sin(robot_cur_state[2] + np.pi/4)])
        # term_2 = (robot_cur_state[2] - self.target_state[2])
        # self.opti.subject_to(term_1 * self.u[0] + term_2 * self.u[1] <= self.slack)

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
    
    def add_cbf_cons_integrate(self, robot_cur_state, robot_params, two_center, obs_state, obs_vertex):
        """ add cons w.r.t other-shaped obstacle """
        # sample points from obstacles and add constraints to the optimal problem
        num_closest_points = 6
        sampled_points = self.robot.get_sampled_points_from_obstacle_vertexes(obs_vertex, num_samples=6)

        cbf_values = []
        obs_states = []

        for i in range(sampled_points.shape[0]):
            obs_cur_point_state = np.array([sampled_points[i][0], sampled_points[i][1], obs_state[2], obs_state[3]])
            cbf = self.cbf(robot_cur_state, robot_params, two_center, obs_cur_point_state)
            cbf_values.append(cbf)
            obs_states.append(obs_cur_point_state)

        cbf_values = np.array(cbf_values)
        obs_states = np.array(obs_states)

        closest_indices = np.argsort(cbf_values)[:num_closest_points]
        for idx in closest_indices:
            obs_state = obs_states[idx]
            cbf = cbf_values[idx]

            lf_cbf, lg_cbf, dt_obs_cbf = self.robot.derive_cbf_gradient(robot_cur_state, robot_params, two_center, obs_state)
            self.opti.subject_to(lf_cbf + (lg_cbf @ self.u)[0, 0] + dt_obs_cbf + self.cbf_gamma * cbf >= 0)

        return np.min(cbf_values)


    def add_cbf_dual_cons(self, robot_cur_state, h_AG, h_BG,
                                mat_A, vec_a, mat_A_dot, vec_a_dot,  # sofa L shape
                                mat_B, vec_b, mat_B_dot, vec_b_dot,  # sofa L shape
                                mat_G, vec_g, mat_G_dot, vec_g_dot,  # one osbtacle
                                lam_A_opt, lam_AG_opt, lam_A_pos_idx, lam_AG_pos_idx,
                                lam_B_opt, lam_BG_opt, lam_B_pos_idx, lam_BG_pos_idx,):
        # paper equation (18a)
        L_dot_AG_1 = - 0.5 * lam_A_opt @ mat_A @ mat_A.T @ self.lam_A_dot.T
        L_dot_AG_2 = - 0.5 * lam_A_opt @ mat_A @ mat_A_dot.T @ lam_A_opt.T
        L_dot_AG_3 = - self.lam_A_dot @ vec_a - lam_A_opt @ vec_a_dot - self.lam_G_dot @ vec_g - lam_AG_opt @ vec_g_dot
        L_dot_AG = L_dot_AG_1 + L_dot_AG_2 + L_dot_AG_3
        self.opti.subject_to(L_dot_AG >= - 1.0 * (h_AG - 0.1))  # paper equation (18a)
        
        L_dot_BG_1 = - 0.5 * lam_B_opt @ mat_B @ mat_B.T @ self.lam_B_dot.T
        L_dot_BG_2 = - 0.5 * lam_B_opt @ mat_B @ mat_B_dot.T @ lam_B_opt.T
        L_dot_BG_3 = - self.lam_B_dot @ vec_b - lam_B_opt @ vec_b_dot - self.lam_G_dot @ vec_g - lam_BG_opt @ vec_g_dot
        L_dot_BG = L_dot_BG_1 + L_dot_BG_2 + L_dot_BG_3
        self.opti.subject_to(L_dot_BG >= - 1.0 * (h_BG - 0.1))  # paper equation (18a)

        # paper cons. (18c)
        self.opti.subject_to( self.lam_A_dot @ mat_A + lam_A_opt @ mat_A_dot + self.lam_G_dot @ mat_G + lam_AG_opt @ mat_G_dot == 0)  
        self.opti.subject_to( self.lam_B_dot @ mat_A + lam_B_opt @ mat_B_dot + self.lam_G_dot @ mat_G + lam_BG_opt @ mat_G_dot == 0)  
        # paper equation (18d)
        for i in range(len(lam_A_pos_idx)):
            self.opti.subject_to(self.lam_A_dot[lam_A_pos_idx[0][i]] >= 0)
        for i in range(len(lam_AG_pos_idx)):
            self.opti.subject_to(self.lam_G_dot[lam_AG_pos_idx[0][i]] >= 0)

        for i in range(len(lam_B_pos_idx)):
            self.opti.subject_to(self.lam_B_dot[lam_B_pos_idx[0][i]] >= 0)
        for i in range(len(lam_BG_pos_idx)):
            self.opti.subject_to(self.lam_G_dot[lam_BG_pos_idx[0][i]] >= 0)
        
        # paper cons. (18e)
        self.opti.subject_to(self.lam_A_dot <= 1e5)
        self.opti.subject_to(self.lam_B_dot <= 1e5)
        self.opti.subject_to(self.lam_G_dot <= 1e5)
        self.opti.subject_to(self.lam_A_dot >= -1e5)
        self.opti.subject_to(self.lam_B_dot >= -1e5)
        self.opti.subject_to(self.lam_G_dot >= -1e5)
    
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

        # the old clfs
        # clf1 = self.add_clf_distance_cons(robot_cur_state, add_slack)
        # clf2 = self.add_clf_theta_cons(robot_cur_state, add_slack)

        # the new clf
        clf3 = self.add_clf_dist_theta_cons(robot_cur_state, add_slack)

        # optimize the qp problem
        try:
            sol = self.opti.solve()
            optimal_control = sol.value(self.u)
            # return optimal_control, clf1, clf2, True
            return optimal_control, clf3, clf3, True
        except:
            print(self.opti.return_status() + ' clf qp')
            return None, None, None, False
        

    def cbf_clf_qp(self, robot_cur_state, dist_AG, dist_BG, 
                        mat_A, vec_a, mat_B, vec_b, mat_G, vec_g,
                        lam_A_opt, lam_AG_opt, lam_A_pos_idx, lam_AG_pos_idx,
                        lam_B_opt, lam_BG_opt, lam_B_pos_idx, lam_BG_pos_idx,
                        add_clf=True, u_ref=None):
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
        # the second qp problem
        if u_ref is None:
            u_ref = np.zeros(self.control_dim)


        R_mat = np.array([[np.cos(robot_cur_state[2]), -np.sin(robot_cur_state[2])], [np.sin(robot_cur_state[2]), np.cos(robot_cur_state[2])]])
        dR_dtheta = np.array([[-np.sin(robot_cur_state[2]), -np.cos(robot_cur_state[2])], [np.cos(robot_cur_state[2]), -np.sin(robot_cur_state[2])]])
        dR_dt_T = dR_dtheta.T * self.u[1]
        dA_dt = mat_A @ dR_dt_T
        dP_dt = np.array([np.cos(robot_cur_state[2]), np.sin(robot_cur_state[2])]) * self.u[0]
        da_dt = mat_A @ ( dR_dt_T @ robot_cur_state[:2] + R_mat.T @ dP_dt)
        dB_dt = mat_B @ dR_dt_T
        db_dt = mat_B @ ( dR_dt_T @ robot_cur_state[:2] + R_mat.T @ dP_dt)
        # obstacle
        dg_dt = np.array([[0, 0, 0, 0]]).reshape(4, 1)  # if the obstacle is static
        dG_dt = np.zeros((4, 2))  

        self.set_optimal_function(u_ref, add_slack=add_clf)
    
        clf = self.add_clf_dist_theta_cons(robot_cur_state, add_slack=add_clf)

        # for plot
        cbf_list = []
        
        # dual constraints
        for i in range(1):
            self.add_cbf_dual_cons(robot_cur_state, dist_AG, dist_BG,
                                        mat_A, vec_a, dA_dt, da_dt, 
                                        mat_B, vec_b, dB_dt, db_dt,
                                        mat_G, vec_g, dG_dt, dg_dt,
                                        lam_A_opt, lam_AG_opt, lam_A_pos_idx, lam_AG_pos_idx,
                                        lam_B_opt, lam_BG_opt, lam_B_pos_idx, lam_BG_pos_idx,)


        # self.add_controls_physical_cons()
        
        cbf_list.append(dist_BG)
        
        # optimize the qp problem
        try:
            sol = self.opti.solve()
            optimal_control = sol.value(self.u)
            if add_clf:
                slack = sol.value(self.slack)
                return optimal_control, cbf_list, clf, slack, True
            else:
                return optimal_control, cbf_list, None, None, True, 
        except:
            print(self.opti.return_status() + ' sdf-cbf with clf')
            return None, cbf_list, None, None, False
