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

        self.opti = ca.Opti()
        # opts_setting = {
        #     'printLevel': 'low',  
        #     'error_on_fail': False,
        #     'expand': True,
        #     'print_time': 0
        # }
        self.opti.solver('ipopt')

        self.u = self.opti.variable(self.control_dim)
        if robot_params['model'] == 'unicycle':
            self.slack1 = self.opti.variable()
            self.slack2 = self.opti.variable()
        elif robot_params['model'] == 'integral':
            self.slack = self.opti.variable()
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
        
    def cbf_clf_qp(self, robot, obs_list, add_clf=True, u_ref=None):
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

