import numpy as np
import casadi as ca
import yaml
from scipy.spatial import ConvexHull
from unicycle_robot_sdf import Unicycle_Robot_Sdf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
        # self.opti = ca.Opti()
        # opts_setting = {
        #         'ipopt.max_iter': 5000,
        #         'ipopt.print_level': 0,
        #         'print_time': 0,
        #         'ipopt.acceptable_tol': 1e-8,
        #         'ipopt.acceptable_obj_change_tol': 1e-6
        #     }
        # self.opti.solver('ipopt', opts_setting)

        # qpoases
        # self.opti = ca.Opti('conic')
        # opts_setting = {
        #     'printLevel': 'low',  
        #     'error_on_fail': False,
        #     'expand': True,
        #     'print_time': 0
        # }
        # self.opti.solver('qpoases', opts_setting)

        # self.u = self.opti.variable(self.control_dim)
        # self.lam_A_dot = self.opti.variable(1, 4)
        # self.lam_B_dot = self.opti.variable(1, 4)
        # self.lam_AG_dot = self.opti.variable(1, 4)
        # self.lam_BG_dot = self.opti.variable(1, 4)

        # self.slack = self.opti.variable()
        # self.obj = None
  
        # approach the desired control and smooth the control
        # self.H = np.diag(self.weight_input)
        # self.R = np.diag(self.smooth_input)

    # def set_optimal_function(self, u_ref, add_slack=True):
    #     """ set the optimal function """
    #     self.obj = (self.u - u_ref).T @ self.H @ (self.u - u_ref)
    #     if add_slack:
    #         # self.obj = self.obj
    #         self.obj = self.obj + self.weight_slack * self.slack ** 2
    #     self.opti.minimize(self.obj)



    
    # def add_clf_dist_theta_cons(self, robot_cur_state, add_slack=False):
    #     """ add clf cons of theta """
    #     # clf3 = self.clf3(robot_cur_state, self.target_state)
    #     # lf_clf3 = self.lf_clf3(robot_cur_state, self.target_state)
    #     # lg_clf3 = self.robot.lg_clf3(robot_cur_state, self.target_state)

    #     # if lg_clf3[0, 1] != 0:
    #     #     if add_slack:
    #     #         self.opti.subject_to(lf_clf3 + (lg_clf3 @ self.u)[0, 0] + self.clf_lambda[1] * clf3 <= self.slack)
    #     #         # self.opti.subject_to(self.opti.bounded(-np.inf, self.slack2, np.inf))
    #     #     else:
    #     #         self.opti.subject_to(lf_clf3 + (lg_clf3 @ self.u)[0, 0] + self.clf_lambda[1] * clf3 <= 0)

    #     # return clf3

    #     # clf from the paper
    #     term_1 = (robot_cur_state[:2] - self.target_state[:2]) @ np.array([np.cos(robot_cur_state[2] + np.pi/4), np.sin(robot_cur_state[2] + np.pi/4)])
    #     term_2 = (robot_cur_state[2] - self.target_state[2])
    #     term_3 = (robot_cur_state - self.target_state) @ (robot_cur_state - self.target_state)
    #     self.opti.subject_to(term_1 * self.u[0] + term_2 * self.u[1] <= -1.0 * term_3 + self.slack)

    #     clf3 = term_3

    #     return clf3

    def dual_qp(self, robot_):
            s = robot_.current_state
            p = s[:2].reshape(2, 1)
            theta = s[2]

            Rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])

            A_new = robot_.A_init @ Rotation_matrix.T
            a_new = robot_.a_init + robot_.A_init @ Rotation_matrix.T @ p
            B_new = robot_.B_init @ Rotation_matrix.T
            b_new = robot_.b_init + robot_.B_init @ Rotation_matrix.T @ p
            G_new = robot_.G_init
            g_new = robot_.g_init
            
            # solve 2 x N QP, N depends on the number of obstacles
            opti = ca.Opti('conic');
            lam_A = opti.variable(1, 4)
            lam_AG = opti.variable(1, 4)
            obj = - 0.25 * lam_A @ A_new @ A_new.T @ lam_A.T - lam_A @ a_new - lam_AG @ g_new
            opti.minimize(-obj)  # max optimization
            opti.subject_to(lam_A @ A_new + lam_AG @ G_new == 0)
            opti.subject_to(lam_A >= 0)
            opti.subject_to(lam_AG >= 0)

            opti.solver('qpoases');  # the options should be alinged with the solver
            sol=opti.solve();
            lam_A_star = sol.value(lam_A).reshape(1, -1)
            lam_AG_star = sol.value(lam_AG).reshape(1, -1)
            lam_A_pos_idx = np.where(lam_A_star[0, :] < 1e-5)
            lam_AG_pos_idx = np.where(lam_AG_star[0, :] < 1e-5)
            dist_square = sol.value(obj)
            dist_AG = np.sqrt(dist_square)

            # # ======================= the second QP problem
            opti = ca.Opti('conic');
            lam_B = opti.variable(1, 4)
            lam_BG = opti.variable(1, 4)
            # the second qp w.r.t. rectangle B
            obj = - 0.25 * lam_B @ B_new @ B_new.T @ lam_B.T - lam_B @ b_new - lam_BG @ g_new
            opti.minimize(-obj)
            opti.subject_to(lam_B @ B_new + lam_BG @ G_new == 0)
            opti.subject_to(lam_B >= 0)
            opti.subject_to(lam_BG >= 0)

            opti.solver('qpoases')
            sol = opti.solve()
            lam_B_star = sol.value(lam_B).reshape(1, -1)
            lam_BG_star = sol.value(lam_BG).reshape(1, -1)
            lam_B_pos_idx = np.where(lam_B_star[0, :] < 1e-5)
            lam_BG_pos_idx = np.where(lam_BG_star[0, :] < 1e-5)
            dist_square_ = sol.value(obj)
            dist_BG = np.sqrt(dist_square_)
    
            print('the AG distance is :', dist_AG)
            print('the BG distance is :', dist_BG)

            return (G_new, g_new, dist_AG, dist_BG, lam_A_star, lam_AG_star, lam_A_pos_idx,
                    lam_AG_pos_idx, lam_B_star, lam_BG_star, lam_B_pos_idx, lam_BG_pos_idx)

    def add_cbf_dual_cons(self, robot_, dual_res, u_ref):
        opti = ca.Opti()
        u = opti.variable(self.control_dim)  # 2 
        slack = opti.variable()  # 1
        lam_dot_i_1 = opti.variable(1, 4)
        lam_dot_i_2 = opti.variable(1, 4)
        lam_dot_j_1 = opti.variable(1, 4)
        lam_dot_j_2 = opti.variable(1, 4)

        A_j, b_j = dual_res[0], dual_res[1]  # obstacle is static
        h_ij_1, h_ij_2 = dual_res[2], dual_res[3]
        lam_star_i_1, lam_star_j_1 = dual_res[4], dual_res[5]
        lam_dot_i_idx = dual_res[6]
        lam_dot_ij_idx = dual_res[7]

        lam_star_i_2, lam_star_j_2 = dual_res[8], dual_res[9]
        lam_dot_i_2_idx = dual_res[10]
        lam_dot_j_2_idx = dual_res[11]
        
        A_i_1_init, b_i_1_init = robot_.A_init, robot_.a_init
        A_i_2_init, b_i_2_init = robot_.B_init, robot_.b_init

        s = robot_.current_state
        p = s[:2].reshape(2, 1)
        Rot = np.array([[np.cos(s[2]), -np.sin(s[2])], [np.sin(s[2]), np.cos(s[2])]])
        dR = np.array([[-np.sin(s[2]), -np.cos(s[2])], [np.cos(s[2]), -np.sin(s[2])]])

        """ set the cost function """
        H_mat = np.diag(self.weight_input)
        obj = (u - u_ref).T @ H_mat @ (u - u_ref)
        obj = obj + self.weight_slack * slack ** 2
        opti.minimize(obj)

        """ set the CLF constraints """
        term_1 = 2*(s[:2] - self.target_state[:2]) @ np.array([np.cos(s[2] + np.pi/4), np.sin(s[2] + np.pi/4)])
        term_2 = 2*(s[2] - self.target_state[2])
        term_3 = (s - self.target_state) @ (s - self.target_state)
        opti.subject_to(term_1 * u[0] + term_2 * u[1] <= -1.0 * term_3 + slack)
        clf = term_3

        """ set the CBF constraints """
        # L_dot between rectangle A and obstacle G
        L_dot_AG_1 = -0.5 * lam_star_j_1 @ (A_j @ A_j.T) @ lam_dot_j_1.T 
        L_dot_AG_2 = 0                          
        L_dot_AG_3 = - lam_dot_i_1 @ (b_i_1_init + A_i_1_init @ Rot.T @ p)
        dARp_dt_1_AG = lam_star_i_1 @ A_i_1_init @ dR.T  @ p * u[1] 
        dARp_dt_2_AG = lam_star_i_1 @ A_i_1_init @ Rot.T @ np.array([np.cos(s[2] + np.pi/4), np.sin(s[2]) + np.pi/4]) * u[0]
        L_dot_AG_4 = - (dARp_dt_2_AG + dARp_dt_1_AG)  
        L_dot_AG_5 = - lam_dot_j_1 @ b_j              
        L_dot_AG_6 = 0                     
        L_dot_AG = L_dot_AG_1 + L_dot_AG_2 + L_dot_AG_3 + L_dot_AG_4 + L_dot_AG_5 + L_dot_AG_6
        opti.subject_to(L_dot_AG >= -0.5 * (h_ij_1 - 0.025**2))          

        # L_dot between rectangle B and obstacle G
        L_dot_BG_1 = -0.5 * lam_star_j_2 @ (A_j @ A_j.T) @ lam_dot_j_2.T    
        L_dot_BG_2 = 0                                                     
        L_dot_BG_3 = - lam_dot_i_2 @ (b_i_2_init + A_i_2_init @ Rot.T @ p) 
        dARp_dt_1_BG = lam_star_i_2 @ A_i_2_init @ dR.T  @ p * u[1]  # robot,
        dARp_dt_2_BG = lam_star_i_2 @ A_i_2_init @ Rot.T @ np.array([np.cos(s[2] + np.pi/4), np.sin(s[2]) + np.pi/4]) * u[0]
        L_dot_BG_4 = - (dARp_dt_2_BG + dARp_dt_1_BG)  
        L_dot_BG_5 = - lam_dot_j_2 @ b_j              
        L_dot_BG_6 = 0
        L_dot_BG = L_dot_BG_1 + L_dot_BG_2 + L_dot_BG_3 + L_dot_BG_4 + L_dot_BG_5 + L_dot_BG_6
        opti.subject_to(L_dot_BG >= -0.5 * (h_ij_2 - 0.025**2))     

        # equality constraints
        eq_AG_1 = lam_dot_i_1 @ (A_i_1_init @ Rot.T)
        eq_AG_2 = lam_star_i_1 @ (A_i_1_init @ dR.T) * u[1]
        eq_AG_3 = lam_dot_j_1 @ (A_j @ Rot.T)
        opti.subject_to(eq_AG_1 + eq_AG_2 + eq_AG_3 == 0)

        eq_BG_1 = lam_dot_i_2 @ (A_i_2_init @ Rot.T)
        eq_BG_2 = lam_star_i_2 @ (A_i_2_init @ dR.T) * u[1]
        eq_BG_3 = lam_dot_j_2 @ (A_j @ Rot.T)
        opti.subject_to(eq_BG_1 + eq_BG_2 + eq_BG_3 == 0)

        # # paper equation (18d)
        for i in range(len(lam_dot_i_idx)):
            opti.subject_to(lam_dot_i_1[lam_dot_i_idx[0][i]] >= 0)

        for i in range(len(lam_dot_ij_idx)):
            opti.subject_to(lam_dot_j_1[lam_dot_ij_idx[0][i]] >= 0)

        for i in range(len(lam_dot_i_2_idx)):
            opti.subject_to(lam_dot_i_2[lam_dot_i_2_idx[0][i]] >= 0)

        for i in range(len(lam_dot_j_2_idx)):
            opti.subject_to(lam_dot_j_2[lam_dot_j_2_idx[0][i]] >= 0)
        
        # # # # paper cons. (18e)
        # opti.subject_to(lam_dot_i_1 <= 1e5)
        # opti.subject_to(lam_dot_j_1 <= 1e5)
        # opti.subject_to(lam_dot_i_2 <= 1e5)
        # opti.subject_to(lam_dot_j_2 <= 1e5)
        # opti.subject_to(lam_dot_i_1 >= -1e5)
        # opti.subject_to(lam_dot_j_1 >= -1e5)
        # opti.subject_to(lam_dot_i_2 >= -1e5)
        # opti.subject_to(lam_dot_j_2 >= -1e5)  
        
        # opti.subject_to(u[0] >= -0.3)
        # opti.subject_to(u[0] <= 0.3)
        # opti.subject_to(u[1] >= -0.2) 
        # opti.subject_to(u[1] <= 0.2)

        # optimize the qp problem
        opts_setting = {
                'ipopt.max_iter': 5000,
                'ipopt.print_level': 0,
                'print_time': 0,
                'ipopt.acceptable_tol': 1e-8,
                'ipopt.acceptable_obj_change_tol': 1e-6
            }
        opti.solver('ipopt', opts_setting)

        try:
            sol = opti.solve()
            optimal_control = sol.value(u)
            slack = sol.value(slack)
            return optimal_control, clf, slack
        except:
            print(opti.return_status() + ' sdf-cbf with clf')
            return None, None, None

    

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
        
   

    def cbf_clf_qp(self, robot_, add_clf=True, u_ref=None):
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

        # solve the dual problems to get the dual variables
        dual_res = self.dual_qp(robot_)

        # dual constraints
        opt_control, clf, slack= self.add_cbf_dual_cons(robot_, dual_res, u_ref)
        
        cbf = []
        cbf.append([dual_res[2], dual_res[3]])

        if opt_control is None:
            return None, cbf, None, None, False
        


        return opt_control, cbf, clf, slack, True
  
