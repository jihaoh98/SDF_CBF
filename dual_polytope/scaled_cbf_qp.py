import numpy as np
import casadi as ca
import yaml
import matplotlib.pyplot as plt 
from robot_sym import Integral_Robot_Sym, Unicycle_Robot_Sym
from derive_cbf_cons import derive_cbf_gradient
from polytopic_robot import Polytopic_Robot
from polytopic_obs import Polytopic_Obs
from math import cos, sin


class Scaled_Cbf:
    def __init__(self, file_name) -> None:
        with open(file_name) as file:
            config = yaml.safe_load(file)
        
        controller_params = config['controller']
        robot_params = config['robot']
        self.robot_model = robot_params['model']

        if robot_params['model'] == 'unicycle':
            self.robot = Unicycle_Robot_Sym()
        elif robot_params['model'] == 'integral':
            self.robot = Integral_Robot_Sym()

        self.state_dim = self.robot.state_dim
        self.control_dim = self.robot.control_dim
        self.target_state = robot_params['target_state']

        if robot_params['model'] == 'unicycle':
            # for distance
            self.clf1 = self.robot.clf1
            self.lf_clf1 = self.robot.lf_clf1
            self.lg_clf1 = self.robot.lg_clf1
            
            # for orientation
            self.clf2 = self.robot.clf2
            self.lf_clf2 = self.robot.lf_clf2
            self.lg_clf2 = self.robot.lg_clf2
        elif robot_params['model'] == 'integral':
            self.clf = self.robot.clf
            self.lf_clf = self.robot.lf_clf
            self.lg_clf = self.robot.lg_clf

        # get the parameter for optimal control
        self.weight_input = controller_params['weight_input']
        self.smooth_input = controller_params['smooth_input']
        self.weight_slack = controller_params['weight_slack']
        self.clf_lambda = controller_params['clf_lambda']
        self.cbf_gamma = controller_params['cbf_gamma']
        self.min_beta = controller_params['min_beta']

        # boundary of control variables
        self.u_max = robot_params['u_max']
        self.u_min = robot_params['u_min']

        self.opti = None
        # opts_setting = {
        #     'printLevel': 'low',  
        #     'error_on_fail': False,
        #     'expand': True,
        #     'print_time': 0
        # }
        # self.opti.solver('ipopt')

        self.u = None
        self.lam_dot_i = None
        self.lam_dot_j = None

        if robot_params['model'] == 'unicycle':
            self.slack1 = None
            self.slack2 = None
            # self.slack = self.opti.variable()

        elif robot_params['model'] == 'integral':
            self.slack = None
        self.obj = None

        # approach the desired control and smooth the control
        self.H = np.diag(self.weight_input)
        self.R = np.diag(self.smooth_input)

    def set_optimal_function(self, u_ref, add_slack=True):
        """ set the optimal function """
        self.obj = (self.u - u_ref).T @ self.H @ (self.u - u_ref)
        if add_slack:
            if self.robot_model == 'unicycle':
                self.obj = self.obj + self.weight_slack[0] * self.slack1 ** 2
                self.obj = self.obj + self.weight_slack[1] * self.slack2 ** 2
            elif self.robot_model == 'integral':
                self.obj = self.obj + self.weight_slack * self.slack ** 2

        self.opti.minimize(self.obj)
        self.opti.subject_to()

    def add_clf_cons_integral(self, robot_cur_state, add_slack=True):
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

    def add_clf_distance_cons(self, robot_cur_state, add_slack=True):
        """ add clf cons of distance """
        clf1 = self.clf1(robot_cur_state, self.target_state)
        lf_clf1 = self.lf_clf1(robot_cur_state, self.target_state)
        lg_clf1 = self.lg_clf1(robot_cur_state, self.target_state)

        if add_slack:
            self.opti.subject_to(lf_clf1 + (lg_clf1 @ self.u)[0, 0] + self.clf_lambda[0] * clf1 <= self.slack1)
            self.opti.subject_to(self.opti.bounded(-np.inf, self.slack1, np.inf))
        else:
            self.opti.subject_to(lf_clf1 + (lg_clf1 @ self.u)[0, 0] + self.clf_lambda[0] * clf1 <= 0)
        
        return clf1
    
    def add_clf_theta_cons(self, robot_cur_state, add_slack=True):
        """ add clf cons of theta """
        clf2 = self.clf2(robot_cur_state, self.target_state)
        lf_clf2 = self.lf_clf2(robot_cur_state, self.target_state)
        lg_clf2 = self.robot.lg_clf2(robot_cur_state, self.target_state)

        if lg_clf2[0, 1] != 0:
            if add_slack:
                self.opti.subject_to(lf_clf2 + (lg_clf2 @ self.u)[0, 0] + self.clf_lambda[1] * clf2 <= self.slack2)
                self.opti.subject_to(self.opti.bounded(-np.inf, self.slack2, np.inf))
            else:
                self.opti.subject_to(lf_clf2 + (lg_clf2 @ self.u)[0, 0] + self.clf_lambda[1] * clf2 <= 0)

        return clf2
    
    def add_controls_physical_cons(self):
        """ add physical constraint of controls """
        self.opti.subject_to(self.opti.bounded(self.u_min, self.u, self.u_max))

    def add_cbf_cons(self, robot: Polytopic_Robot, obs: Polytopic_Obs):
        """ add cons w.r.t a polytopic obstacle """
        beta, dbeta_dx, dbeta_dp_o = derive_cbf_gradient(robot, obs)
        if beta is None:
            return
        
        g = np.array([[1, 0], [0, 1], [0, 0]])
        if self.robot_model == 'unicycle':
            theta = robot.cur_state[2]
            g = np.array([[cos(theta), 0], [sin(theta), 0], [0, 1]])

        cbf = beta - self.min_beta
        lf_cbf = 0.0
        lg_cbf = dbeta_dx.T @ g 
        dt_obs_cbf = (dbeta_dp_o.T @ obs.vel.reshape(-1, 1))[0, 0]
        self.opti.subject_to(lf_cbf + (lg_cbf @ self.u)[0, 0] + dt_obs_cbf + self.cbf_gamma * cbf >= 0)

        return cbf
    
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
        if self.robot_model == 'unicycle':
            clf1 = self.add_clf_distance_cons(robot_cur_state, add_slack)
            clf2 = self.add_clf_theta_cons(robot_cur_state, add_slack)
        elif self.robot_model == 'integral':
            clf = self.add_clf_cons_integral(robot_cur_state, add_slack)

        # self.add_controls_physical_cons()

        # optimize the qp problem
        try:
            sol = self.opti.solve()
            optimal_control = sol.value(self.u)
            if self.robot_model == 'unicycle':
                return optimal_control, clf1, clf2, True
            elif self.robot_model == 'integral':
                return optimal_control, clf, True
        except:
            print(self.opti.return_status() + ' clf-qp')
            if self.robot_model == 'unicycle':
                return None, None, None, False
            elif self.robot_model == 'integral':
                return None, None, False
        
    def dual_cbf_clf_qp(self, robot, obs_list, add_clf=True, u_ref=None):
        """
        This is a function to calculate the optimal control for the polytopic-shaped robot and robots
        Returns:
            optimal control u
        """
        if u_ref is None:
            u_ref = np.zeros(self.control_dim)
        self.set_optimal_function(u_ref, add_slack=add_clf)

        robot_cur_state = robot.cur_state
        if self.robot_model == 'unicycle':
            clf1, clf2 = None, None
            if add_clf:
                clf1 = self.add_clf_distance_cons(robot_cur_state, add_slack=True)
                clf2 = self.add_clf_theta_cons(robot_cur_state, add_slack=True)
        elif self.robot_model == 'integral':
            clf = None
            if add_clf:
                clf = self.add_clf_cons_integral(robot_cur_state, add_slack=True)

        cbf_list = []
        for obs in obs_list:
            cbf = self.add_cbf_cons(robot, obs)
            cbf_list.append(cbf)
    
        self.add_controls_physical_cons()
        
        print('start solve')
        # optimize the qp problem
        try:
            sol = self.opti.solve()
            optimal_control = sol.value(self.u)
            if self.robot_model == 'unicycle':
                if add_clf:
                    slack1 = sol.value(self.slack1)
                    slack2 = sol.value(self.slack2)
                    return optimal_control, cbf_list, clf1, clf2, slack1, slack2, True
                else:
                    return optimal_control, cbf_list, None, None, None, None, True
            elif self.robot_model == 'integral':
                if add_clf:
                    slack = sol.value(self.slack)
                    return optimal_control, cbf_list, clf, slack, True
                else:
                    return optimal_control, cbf_list, None, None, True
        except:
            print(self.opti.return_status() + ' scaled cbf-clf-qp')
            if self.robot_model == 'unicycle':
                return None, cbf_list, None, None, None, None, False
            elif self.robot_model == 'integral':
                return None, cbf_list, None, None, False


    def dual_qp(self, robot_):
        s = robot_.cur_state
        p = s[:2].reshape(2, 1)
        theta = s[2]

        Rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        A_new = robot_.A0 @ Rotation_matrix.T
        b_new = robot_.b0 + robot_.A0 @ Rotation_matrix.T @ p
        G_new = robot_.G0
        g_new = robot_.g0
        
        # solve 2 x N QP, N depends on the number of obstacles
        opti = ca.Opti('conic');
        lam_A = opti.variable(1, 4)
        lam_AG = opti.variable(1, 4)
        obj = - 0.25 * lam_A @ A_new @ A_new.T @ lam_A.T - lam_A @ b_new - lam_AG @ g_new
        opti.minimize(-obj)  # max optimization
        opti.subject_to()
        opti.subject_to(lam_A @ A_new + lam_AG @ G_new == 0)
        opti.subject_to(lam_A >= 0)
        opti.subject_to(lam_AG >= 0)

        opti.solver('qpoases');  # the options should be alinged with the solver
        sol=opti.solve();
        lam_A_star = sol.value(lam_A).reshape(1, -1)
        lam_AG_star = sol.value(lam_AG).reshape(1, -1)
        lam_A_pos_idx = np.where(lam_A_star[0, :] < 1e-5)
        lam_AG_pos_idx = np.where(lam_AG_star[0, :] < 1e-5)
        lam_A_pos_idx = lam_A_pos_idx[0].tolist()
        lam_AG_pos_idx = lam_AG_pos_idx[0].tolist()
        dist_square = sol.value(obj)
        dist_AG = np.sqrt(dist_square)

        print('the AG distance is :', np.sqrt(dist_square))

        return (G_new, g_new, dist_AG, lam_A_star, lam_AG_star, lam_A_pos_idx, lam_AG_pos_idx)

    def add_cbf_dual_cons(self, robot_, dual_res, u_ref):

        A_j, b_j = dual_res[0], dual_res[1]  # obstacle is static
        h_ij = dual_res[2]
        lam_star_i = dual_res[3]
        lam_star_j = dual_res[4]
        lam_dot_i_idx = dual_res[5]
        lam_dot_ij_idx = dual_res[6]
        
        A_i_init, b_i_init = robot_.A0, robot_.b0

        s = robot_.cur_state
        p = s[:2].reshape(2, 1)
        Rot = np.array([[np.cos(s[2]), -np.sin(s[2])], [np.sin(s[2]), np.cos(s[2])]])
        dR = np.array([[-np.sin(s[2]), -np.cos(s[2])], [np.cos(s[2]), -np.sin(s[2])]])

        """ set the CBF constraints """
        # L_dot between rectangle A and obstacle G
        L_dot_AG_1 = -0.5 * lam_star_j @ (A_j @ A_j.T) @ self.lam_dot_j.T 
        L_dot_AG_2 = 0                          
        L_dot_AG_3 = - self.lam_dot_i @ (b_i_init + A_i_init @ Rot.T @ p)
        dARp_dt_1_AG = lam_star_i @ A_i_init @ dR.T  @ p * self.u[1] 
        dARp_dt_2_AG = lam_star_i @ A_i_init @ Rot.T @ np.array([np.cos(s[2]), np.sin(s[2])]) * self.u[0]
        L_dot_AG_4 = - (dARp_dt_2_AG + dARp_dt_1_AG)  
        L_dot_AG_5 = - self.lam_dot_j @ b_j              
        L_dot_AG_6 = 0                     
        L_dot_AG = L_dot_AG_1 + L_dot_AG_2 + L_dot_AG_3 + L_dot_AG_4 + L_dot_AG_5 + L_dot_AG_6
        self.opti.subject_to(L_dot_AG >= -2.0 * (h_ij - 0.025**2))          

        # equality constraints
        eq_AG_1 = self.lam_dot_i @ (A_i_init @ Rot.T)
        eq_AG_2 = lam_star_i @ (A_i_init @ dR.T) * self.u[1]
        eq_AG_3 = self.lam_dot_j @ (A_j @ Rot.T)
        self.opti.subject_to(eq_AG_1 + eq_AG_2 + eq_AG_3 == 0)

        # paper equation (18d)
        for i in range(len(lam_dot_i_idx)):
            self.opti.subject_to(self.lam_dot_i[lam_dot_i_idx[i]] >= 0)

        for i in range(len(lam_dot_ij_idx)):
            self.opti.subject_to(self.lam_dot_j[lam_dot_ij_idx[i]] >= 0)


        # # # # # paper cons. (18e)
        self.opti.subject_to(self.lam_dot_i <= 1e3)
        self.opti.subject_to(self.lam_dot_j <= 1e3)
        self.opti.subject_to(self.lam_dot_i >= -1e3)
        self.opti.subject_to(self.lam_dot_j >= -1e3)  

        self.opti.subject_to(self.u[0] >= -0.3)
        self.opti.subject_to(self.u[0] <= 0.3)
        self.opti.subject_to(self.u[1] >= -0.2) 
        self.opti.subject_to(self.u[1] <= 0.2)

        # optimize the qp problem
        opts_setting = {
                'ipopt.max_iter': 5000,
                'ipopt.print_level': 0,
                'print_time': 0,
                'ipopt.acceptable_tol': 1e-8,
                'ipopt.acceptable_obj_change_tol': 1e-6
            }
        self.opti.solver('ipopt', opts_setting)

        try:
            sol = self.opti.solve()
            optimal_control = sol.value(self.u)
            lam_dot_i = sol.value(self.lam_dot_i)
            lam_dot_j = sol.value(self.lam_dot_j)
            return optimal_control, lam_dot_i, lam_dot_j
        except:
            print(self.opti.return_status() + ' sdf-cbf with clf')
            return None, None, None
        
            
    def cbf_clf_qp(self, robot_, add_clf=True, u_ref=None):
        """
        This is a function to calculate the optimal control for the polytopic-shaped robot and robots
        Returns:
            optimal control u
        """
        dual_res = self.dual_qp(robot_)
        lam_i = dual_res[3]
        lam_j = dual_res[4]
        print('lam_i:', lam_i)
        print('lam_j:', lam_j)
        if u_ref is None:
            u_ref = np.zeros(self.control_dim)
        
        opti = ca.Opti()
        u = opti.variable(self.control_dim)
        slack1 = opti.variable()
        slack2 = opti.variable()
        self.u = u
        self.opti = opti
        self.slack1 = slack1
        self.slack2 = slack2
        self.lam_dot_i = opti.variable(1, 4)
        self.lam_dot_j = opti.variable(1, 4)
        
        self.set_optimal_function(u_ref, add_slack=add_clf)

        robot_cur_state = robot_.cur_state
        if self.robot_model == 'unicycle':
            clf1, clf2 = None, None
            if add_clf:
                clf1 = self.add_clf_distance_cons(robot_cur_state, add_slack=True)
                clf2 = self.add_clf_theta_cons(robot_cur_state, add_slack=True)
        elif self.robot_model == 'integral':
            clf = None
            if add_clf:
                clf = self.add_clf_cons_integral(robot_cur_state, add_slack=True)

        cbf_list = []
        opt_control, lam_dot_i, lam_dot_j = self.add_cbf_dual_cons(robot_, dual_res, u_ref)
        print('lam_dot_i:', lam_dot_i)
        print('lam_dot_j:', lam_dot_j)

        
        cbf_list.append(dual_res[2])

        if np.isnan(dual_res[2]):
            print('nan')
            
            
        if opt_control is (None, None):
            return None, cbf_list, None, None, None, None, False
        
        return opt_control, cbf_list, clf1, clf2, lam_i, lam_j, True




