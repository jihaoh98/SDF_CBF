import numpy as np
import casadi as ca
import integral_robot_sdf
import yaml
import time


class Integral_Sdf_Cbf_Clf:
    def __init__(self, file_name) -> None:
        """ init the optimal problem with clf-cbf-qp """
        with open(file_name) as file:
            config = yaml.safe_load(file)

        controller_params = config['controller']
        robot_params = config['robot']

        # init robot
        self.robot = integral_robot_sdf.Integral_Robot_Sdf(robot_params)  # define clf, cbf, dynamics
        self.state_dim = self.robot.state_dim
        self.control_dim = self.robot.control_dim
        self.target_state = robot_params['target_state']
        self.sensor_range = robot_params['sensor_range']
        self.robot_radius = robot_params['radius']

        # initialize CLF 
        self.clf = self.robot.clf
        self.lf_clf = self.robot.lf_clf
        self.lg_clf = self.robot.lg_clf

        # initialize CBF
        self.cbf = self.robot.cbf
        self.lf_cbf = self.robot.lf_cbf
        self.lg_cbf = self.robot.lg_cbf
        self.dt_cbf = self.robot.dt_cbf
        self.dx_cbf = self.robot.dx_cbf

        # get the parameter for optimal control
        self.weight_input = controller_params['weight_input']
        self.smooth_input = controller_params['smooth_input']
        self.weight_slack = controller_params['weight_slack']
        self.clf_lambda = controller_params['clf_lambda']
        self.cbf_gamma = controller_params['cbf_gamma']

        # boundary of control variables
        self.u_max = robot_params['u_max']
        self.u_min = robot_params['u_min']

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

    def add_cdf_cbf_cons(self, robot_state, dist_input, grad_input):
        """ add cons w.r.t cdf obstacles """
        cbf = dist_input
        lf_cbf, lg_cbf, dt_cbf = self.robot.derive_cdf_cbf_derivative(robot_state, dist_input, grad_input)
        dt_cbf = 0
        self.opti.subject_to(lf_cbf + (lg_cbf @ self.u)[0, 0] + dt_cbf + self.cbf_gamma * cbf >= 0)
        return cbf, grad_input

    def add_cir_cbf_cons(self, robot_state_cbf, cir_obs_state):
        """ add cons w.r.t circle obstacle """
        cbf = self.cbf(robot_state_cbf, cir_obs_state)
        lf_cbf = self.lf_cbf(robot_state_cbf, cir_obs_state)
        lg_cbf = self.lg_cbf(robot_state_cbf, cir_obs_state)
        dt_cbf = self.dt_cbf(robot_state_cbf, cir_obs_state)
        dx_cbf = self.dx_cbf(robot_state_cbf, cir_obs_state)

        self.opti.subject_to(lf_cbf + (lg_cbf @ self.u)[0, 0] + dt_cbf + self.cbf_gamma * cbf >= 0)
        return cbf, dx_cbf

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

        # result
        result = lambda: None
        result.clf = clf

        # optimize the qp problem
        try:
            start_time = time.time()
            sol = self.opti.solve()
            end_time = time.time()
            optimal_control = sol.value(self.u)

            result.u = optimal_control
            result.time = end_time - start_time
            result.feas = True
        except:
            print(self.opti.return_status() + ' clf qp')
            result.u = None
            result.time = None
            result.feas = False

        return result

    def cbf_clf_cdf_qp(self, robot_cur_state, dist_input, grad_input, add_clf=True, u_ref=None):
        if u_ref is None:
            u_ref = np.zeros(self.control_dim)

        clf = None
        if add_clf:
            clf = self.add_clf_cons(robot_cur_state, add_slack=add_clf)
        # add cbf constraints for each cdf obstacles
        cdf_cbf_list = None
        cdf_dx_cbf_list = None
        # robot_state_cbf = np.hstack((robot_cur_state, np.array([self.robot_radius])))
        if dist_input is not None and grad_input is not None:
            cdf_cbf_list = []
            cdf_dx_cbf_list = []
            for i in range(len(dist_input)):
                cbf, dx_cbf = self.add_cdf_cbf_cons(robot_cur_state, dist_input[i], grad_input[i])
                cdf_cbf_list.append(cbf)
                cdf_dx_cbf_list.append(dx_cbf)

        self.add_controls_physical_cons()

        # result
        result = lambda: None
        result.clf = clf
        result.cdf_cbf_list = cdf_cbf_list
        result.cdf_dx_cbf_list = cdf_dx_cbf_list

        # optimize the qp problem
        try:
            start_time = time.time()
            sol = self.opti.solve()
            end_time = time.time()
            optimal_control = sol.value(self.u)

            result.u = optimal_control
            result.time = end_time - start_time
            result.feas = True

            if add_clf:
                slack = sol.value(self.slack)
                result.slack = slack
            else:
                result.slack = None
        except:
            print(self.opti.return_status() + ' sdf-cbf with clf')
            result.u = None
            result.time = None
            result.feas = False
            result.slack = None

        return result

    def cbf_clf_qp(self, robot_cur_state, cir_obs_states=None, add_clf=True, u_ref=None):
        """
        This is a function to calculate the optimal control for the robot w.r.t circular-shaped obstacles
        Args:
            robot_cur_state: [x, y, theta] np.array(3, )
            cir_obs_state: [ox, oy, ovx, ovy, o_radius] list for circle obstacle state
            robot_state_cbf: [x, y, theta, radius] np.array(4, )
        Returns:
            optimal control u
        """
        if u_ref is None:
            u_ref = np.zeros(self.control_dim)
        self.set_optimal_function(u_ref, add_slack=add_clf)

        clf = None
        if add_clf:
            clf = self.add_clf_cons(robot_cur_state, add_slack=add_clf)

        # add cbf constraints for each circular-shaped obstacles
        cir_cbf_list = None
        cir_dx_cbf_list = None
        robot_state_cbf = np.hstack((robot_cur_state, np.array([self.robot_radius])))
        if cir_obs_states is not None:
            cir_cbf_list = []
            cir_dx_cbf_list = []
            for cir_obs_state in cir_obs_states:
                cbf, dx_cbf = self.add_cir_cbf_cons(robot_state_cbf, cir_obs_state)
                cir_cbf_list.append(cbf)
                cir_dx_cbf_list.append(dx_cbf)

        self.add_controls_physical_cons()

        # result
        result = lambda: None
        result.clf = clf
        result.cir_cbf_list = cir_cbf_list
        result.cir_dx_cbf_list = cir_dx_cbf_list

        # optimize the qp problem
        try:
            start_time = time.time()
            sol = self.opti.solve()
            end_time = time.time()
            optimal_control = sol.value(self.u)

            result.u = optimal_control
            result.time = end_time - start_time
            result.feas = True

            if add_clf:
                slack = sol.value(self.slack)
                result.slack = slack
            else:
                result.slack = None
        except:
            print(self.opti.return_status() + ' sdf-cbf with clf')
            result.u = None
            result.time = None
            result.feas = False
            result.slack = None

        return result
