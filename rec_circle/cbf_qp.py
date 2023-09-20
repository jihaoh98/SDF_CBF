import numpy as np
import casadi as ca


class Sdf_Cbf:
    def __init__(self, system, params) -> None:
        # dimension
        self.robot_system = system
        self.state_dim = system.state_dim 
        self.control_dim = system.control_dim

        # initialize CLF
        self.clf = system.clf
        self.lf_clf = system.lf_clf
        self.lg_clf = system.lg_clf

        # initialize CBF
        self.cbf = system.cbf

        # get the parameter for optimal control
        self.weight_input = params['weight_input']
        self.smooth_input = params['smooth_input']
        self.weight_slack = params['weight_slack']

        self.clf_lambda = params['clf_lambda']
        self.cbf_gamma = params['cbf_gamma']

        self.target_state = params['target_state']
        self.sensor_range = params['sensor_range']

        # boundary of control variables
        self.u_max = params['u_max']
        self.u_min = params['u_min']

        # for optimal-decay cbf
        # self.w0 = params['w0']
        # self.weight_w = params['weight_w']

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

    def cbf_clf_qp(self, robot_cur_state, obstacle_state, add_clf=True, u_ref=None):
        """
        This is a function to calculate the optimal control
        Args:
            robot_cur_state: [x, y] np.array(2, )
            obstacle_state: [ox, oy, ovx, ovy] np.array(4, )
        Returns:
            optimal control
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
            self.opti.subject_to(self.opti.bounded(-np.inf, self.slack, np.inf))
            clf = self.robot_system.clf(robot_cur_state, self.target_state)
            lf_clf = self.robot_system.lf_clf(robot_cur_state, self.target_state)
            lg_clf = self.robot_system.lg_clf(robot_cur_state, self.target_state)

            self.opti.subject_to(lf_clf + (lg_clf @ self.u)[0, 0] + self.clf_lambda * clf <= self.slack)

        # cbf constraint
        cbf = self.cbf(robot_cur_state, obstacle_state)
        lf_cbf, lg_cbf, dt_obs_cbf = self.robot_system.derive_cbf_gradient(robot_cur_state, obstacle_state)
        self.opti.subject_to(lf_cbf + (lg_cbf @ self.u)[0, 0] + dt_obs_cbf + self.cbf_gamma * cbf >= 0)

        # set the boundary constraints
        self.opti.subject_to(self.opti.bounded(self.u_min, self.u, self.u_max))
        
        # optimize the qp problem
        try:
            sol = self.opti.solve()
            optimal_control = sol.value(self.u)
            if add_clf:
                slack = sol.value(self.slack)
                return optimal_control, clf, cbf, slack, True
            else:
                return optimal_control, None, cbf, None, True
        except:
            print(self.opti.return_status() + ' sdf-cbf with clf')
            return None, None, None, None, False







