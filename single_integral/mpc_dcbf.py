import time
import yaml
import statistics
import numpy as np
import casadi as ca

from mpc_render import Mpc_Render


class Mpc_Controller:
    def __init__(self, file_name) -> None:
        with open(file_name) as file:
            config = yaml.safe_load(file)

        robot_params = config["robot"]
        controller_params = config["controller"]
        obs_params = config["cir_obs"]

        # robot parameters
        self.robot_radius = robot_params["radius"]
        self.vx_min = robot_params["vx_boundary"][0]
        self.vx_max = robot_params["vx_boundary"][1]
        self.vy_min = robot_params["vy_boundary"][0]
        self.vy_max = robot_params["vy_boundary"][1]

        self.init_state = np.array(robot_params["init_state"])
        self.target = np.array(robot_params["target"])
        self.goal_threshold = robot_params["goal_threshold"]
        self.f = lambda x_, u_: ca.horzcat(*[u_[0], u_[1]])

        # controller parameters
        self.N = controller_params["N"]
        self.dt = controller_params["dt"]
        self.margin = controller_params["margin"]
        self.gamma = controller_params["gamma"]

        # obstacle parameters
        self.obs_num = len(obs_params["obs_states"])
        self.obs_states = np.array(obs_params["obs_states"])

        # opti stack
        self.opti = ca.Opti()
        self.states = self.opti.variable(self.N + 1, 2)
        self.controls = self.opti.variable(self.N, 2)

        self.opt_init_state = self.opti.parameter(2)
        self.desired_traj = self.opti.parameter(self.N + 1, 2)
        self.obj = None

        self.vx = self.controls[:, 0]
        self.vy = self.controls[:, 1]

        # init values
        self.states_predict = np.zeros((self.N + 1, 2))
        self.controls_predict = np.zeros((self.N, 2))

        # ipopt solver
        opts_setting = {
            "ipopt.max_iter": 100,
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.acceptable_tol": 1e-8,
            "ipopt.acceptable_obj_change_tol": 1e-6,
        }
        self.opti.solver("ipopt", opts_setting)

        self.render = Mpc_Render(robot_params, obs_params, self.dt)

    def optimal_control(self):
        """solve the optimal control problem"""
        process_time = []
        x_optimal_list = []
        x_optimal_predict_list = []
        u_optimal_list = []

        obs_states_t = [[] for _ in range(self.obs_num)]
        obs_cur_states = self.obs_states.copy()

        x0 = self.init_state.copy()
        x_optimal_list.append(x0)
        x_optimal_predict_list.append(np.zeros((self.N, 2)))
        for i in range(self.obs_num):
            obs_states_t[i].append(obs_cur_states[i].copy())

        for k in range(500 - self.N):
            desired_states = self.get_desired_traj(x0, self.target)
            mpc_start = time.time()
            result = self.solve_mpc(x0, desired_states, obs_cur_states)
            mpc_end = time.time()
            process_time.append((mpc_end - mpc_start) * 1000)

            x_predict, u_optimal = result.x, result.u
            x0 = x_predict[0, :]
            x_optimal_list.append(x0)
            x_optimal_predict_list.append(x_predict)
            u_optimal_list.append(u_optimal[0, :])

            for i in range(self.obs_num):
                obs_cur_states[i][0:2] = (
                    obs_cur_states[i][0:2] + obs_cur_states[i][2:4] * self.dt
                )
                obs_states_t[i].append(obs_cur_states[i].copy())

            if np.linalg.norm(x0[0:2] - self.target[0:2], 2) <= self.goal_threshold:
                print("The robot has reached its goal in {} time!".format(k))
                break

        print("Maxinum_time:", max(process_time))
        print("Minimum_time:", min(process_time))
        print("Median_time:", statistics.median(process_time))
        print("Average_time:", statistics.mean(process_time))

        self.render.cir_render(
            x_optimal_list, x_optimal_predict_list, self.target, obs_states_t
        )

    def get_desired_traj(self, cur_state, goal_state):
        """get the desired trajectory"""
        desired_states = np.zeros((self.N + 1, 2))

        desired_states[0, :] = cur_state.copy()
        for i in range(1, self.N + 1):
            desired_states[i, 0] = desired_states[i - 1, 0] + 0.5 * self.dt
            if desired_states[i, 0] > goal_state[0]:
                desired_states[i, 0] = goal_state[0]
            desired_states[i, 1] = goal_state[1]

        return desired_states

    def set_objective_function(self):
        self.obj = 0
        R = np.diag([0.1, 0.02])
        for i in range(self.N):
            Q = np.diag([1.0 + 0.05 * i, 1.0 + 0.05 * i])
            self.obj += (
                (self.states[i + 1, :] - self.desired_traj[i + 1, :])
                @ Q
                @ (self.states[i + 1, :] - self.desired_traj[i + 1, :]).T
            )
            self.obj += self.controls[i, :] @ R @ self.controls[i, :].T

        self.opti.minimize(self.obj)
        self.opti.subject_to()

    def add_states_constraints(self):
        """add states constraints"""
        self.opti.subject_to(self.states[0, :] == self.opt_init_state.T)
        for i in range(self.N):
            x_next = (
                self.states[i, :]
                + self.f(self.states[i, :], self.controls[i, :]) * self.dt
            )
            self.opti.subject_to(self.states[i + 1, :] == x_next)

    def add_controls_constraints(self):
        """add physical constraints of controls"""
        self.opti.subject_to(self.opti.bounded(self.vx_min, self.vx, self.vx_max))
        self.opti.subject_to(self.opti.bounded(self.vy_min, self.vy, self.vy_max))

    def add_collision_avoidance_constraints(self, obs_states):
        """add collision avoidance constraints through cbf w.r.t one obstacle"""
        def h(r, ob):
            dist = (
                ca.sqrt((r[0] - ob[0]) ** 2 + (r[1] - ob[1]) ** 2)
                - self.robot_radius
                - ob[4]
                - self.margin
            )
            return dist

        for i in range(self.N):
            self.opti.subject_to(
                h(self.states[i + 1, :], obs_states[i + 1, :])
                >= (1 - self.gamma) * h(self.states[i, :], obs_states[i, :])
            )

    def predict_obs_trajectory(self, obs_state):
        predict_trajectory = obs_state.copy()
        cur_state = obs_state.copy()
        for _ in range(self.N):
            cur_state = self.move_with_omni_vel(cur_state, cur_state[2], cur_state[3])
            predict_trajectory = np.vstack((predict_trajectory, cur_state))

        return predict_trajectory

    def move_with_omni_vel(self, state, vx, vy):
        """move with the input velocity"""
        res_state = state.copy()

        res_state[0] += vx * self.dt
        res_state[1] += vy * self.dt
        res_state[2] = vx
        res_state[3] = vy
        return res_state

    def solve_mpc(self, cur_state, desired_states, obs_states):
        """
        solve the mpc optimal problem
        Args:
            cur_state: [x, y]
            desired_states: [[x, y] ... ]
            obs_states: [[x. y, vx, vy, radius] ...]
        Returns:
            result:
            x: predicted robot states
            u: predicted controls
            time: computation time
            feas: feasible or not
        """
        self.set_objective_function()
        self.opti.set_value(self.opt_init_state, cur_state)
        self.opti.set_value(self.desired_traj, desired_states)

        self.add_states_constraints()
        self.add_controls_constraints()

        obs_num = len(obs_states)
        for i in range(obs_num):
            cur_obs_state = obs_states[i].copy()
            pre_obs_states = self.predict_obs_trajectory(cur_obs_state)
            self.add_collision_avoidance_constraints(pre_obs_states)

        # set the initial value of optimal variables
        self.opti.set_initial(self.states, self.states_predict)
        self.opti.set_initial(self.controls, self.controls_predict)

        result = lambda: None
        try:
            start_time = time.time()
            sol = self.opti.solve()
            end_time = time.time()
            self.states_predict = sol.value(self.states)
            self.controls_predict = sol.value(self.controls)

            # update the initial condition
            for i in range(self.N):
                self.states_predict[i, :] = self.states_predict[i + 1, :]
            self.states_predict[self.N, :] = self.states_predict[self.N - 1, :]
            for i in range(self.N - 1):
                self.controls_predict[i, :] = self.controls_predict[i + 1, :]
            self.controls_predict[self.N - 1, :] = np.array([0, 0])

            result.x = self.states_predict[1:, :]
            result.u = self.controls_predict
            result.time = end_time - start_time
            result.feas = True

        except:
            self.opti.debug.show_infeasibilities()
            print(self.opti.return_status() + " udeer mpc test")
            result.x = None
            result.u = None
            result.time = None
            result.feas = False

        return result


if __name__ == "__main__":
    mpc_controller = Mpc_Controller("test.yaml")
    mpc_controller.optimal_control()
