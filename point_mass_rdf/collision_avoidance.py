import numpy as np
from integral_sdf_qp import Integral_Sdf_Cbf_Clf
import time
import yaml
import obs
import statistics
import os
import torch
from cdf import CDF2D
from primitives2D_torch import Circle
from render_show import Render_Animation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
print("CURRENT_DIR:", CURRENT_DIR)


class Collision_Avoidance:
    def __init__(self, file_name, case_flag) -> None:
        """ collision avoidance with obstacles """
        with open(file_name) as file:
            config = yaml.safe_load(file)

        robot_params = config['robot']
        controller_params = config['controller']
        self.cbf_qp = Integral_Sdf_Cbf_Clf(file_name)

        # init robot state
        self.robot_init_state = np.array(robot_params['initial_state'])
        self.robot_cur_state = np.copy(self.robot_init_state)
        self.robot_target_state = np.array(robot_params['target_state'])
        self.destination_margin = robot_params['destination_margin']
        self.margin = robot_params['margin']
        self.l = robot_params['l']

        "init obstacle, if no data, return None"
        obs_params = None
        if case_flag == 1:
            pass
        elif case_flag == 2:
            pass
        elif case_flag == 3:
            obs_params = config.get('eef_static_rdf_clf_list')
        elif case_flag == 4:
            pass
        elif case_flag == 5:
            obs_params = config.get('static_rdf_clf_list')

        cir_obs_params = config.get('cir_obstacle_list')
        cdf_sta_obs_params = config.get('cdf_sta_obstacle_list')
        cdf_dyn_obs_params = config.get('cdf_dyn_obstacle_list')

        self.cir_obs_states_list = None
        self.cdf_dyn_obs_states_list = None
        self.cdf_dyn_obs_num = 0
        self.cdf_dyn_obs_center_list = []

        # if cir_obs_params is not None:
        #     self.cir_obs_num = len(cir_obs_params['obs_states'])
        #     self.cir_obs_list = [None for i in range(self.cir_obs_num)]
        #     for i in range(self.cir_obs_num):
        #         self.cir_obs_list[i] = obs.Circle_Obs(
        #             index=i,
        #             radius=cir_obs_params['obs_radii'][i],
        #             center=cir_obs_params['obs_states'][i],
        #             vel=cir_obs_params['obs_vel'][i],
        #             mode=cir_obs_params['modes'][i],
        #         )
        #
        #     # get cir_obstacles' center position and velocity as well as radius
        #     self.cir_obs_init_states_list = [
        #         self.cir_obs_list[i].get_current_state()
        #         for i in range(self.cir_obs_num)
        #     ]
        #     self.cir_obs_states_list = np.copy(self.cir_obs_init_states_list)

        # if cdf_sta_obs_params is not None:
        #     self.cdf_sta_obs_num = len(cdf_sta_obs_params['obs_states'])
        #     self.cdf_sta_obs_list = [None for i in range(self.cdf_sta_obs_num)]
        #     for i in range(self.cdf_sta_obs_num):
        #         self.cdf_sta_obs_list[i] = obs.Cdf_Obs(
        #             index=i,
        #             radius=cdf_sta_obs_params['obs_radii'][i],
        #             center=cdf_sta_obs_params['obs_states'][i],
        #             vel=cdf_sta_obs_params['obs_vel'][i],
        #             mode=cdf_sta_obs_params['modes'][i],
        #         )

        # if cdf_dyn_obs_params is not None:
        #     self.dyn_obstacle_gradient_filed = []
        #     self.cdf_dyn_obs_num = len(cdf_dyn_obs_params['obs_states'])
        #     self.cdf_dyn_obs_list = [None for i in range(self.cdf_dyn_obs_num)]
        #     for i in range(self.cdf_dyn_obs_num):
        #         self.cdf_dyn_obs_list[i] = obs.Cdf_Obs(
        #             index=i,
        #             radius=cdf_dyn_obs_params['obs_radii'][i],
        #             center=cdf_dyn_obs_params['obs_states'][i],
        #             vel=cdf_dyn_obs_params['obs_vel'][i],
        #             mode=cdf_dyn_obs_params['modes'][i],
        #         )

        if obs_params is not None:
            self.sdf_sta_obs_num = len(obs_params['obs_states'])
            self.sdf_sta_obs_list = [None for i in range(self.sdf_sta_obs_num)]
            for i in range(self.sdf_sta_obs_num):
                self.sdf_sta_obs_list[i] = obs.Sdf_Obs(
                    index=i,
                    radius=obs_params['obs_radii'][i],
                    center=obs_params['obs_states'][i],
                    vel=obs_params['obs_vel'][i],
                    mode=obs_params['modes'][i],
                )

        # controller
        self.T = controller_params['Tmax']
        self.step_time = controller_params['step_time']
        self.time_steps = int(np.ceil(self.T / self.step_time))
        self.terminal_time = self.time_steps

        # storage
        self.xt = np.zeros((2, self.time_steps + 1))
        self.ut = np.zeros((2, self.time_steps))
        self.clft = np.zeros((1, self.time_steps))
        self.slackt = np.zeros((1, self.time_steps))

        "storage for circle obstacles"
        if cir_obs_params is not None:
            self.cir_obstacle_state_t = None
            self.cir_obs_cbf_t = None
            self.cir_obs_dx_cbf_t = None
            self.cir_obstacle_state_t = np.zeros((self.cir_obs_num, 5, self.time_steps + 1))
            self.cir_obs_cbf_t = np.zeros((self.cir_obs_num, self.time_steps))
            self.cir_obs_dx_cbf_t = np.zeros((self.cir_obs_num, 2, self.time_steps))
            self.cir_obs_dot_cbf_t = np.zeros((self.cir_obs_num, 2, self.time_steps))

        "storage for static cdf obstacles"
        if cdf_sta_obs_params is not None:
            self.cdf_obs_cbf_t = np.zeros((self.cdf_sta_obs_num, 1, self.time_steps))
            self.cdf_obs_dx_cbf_t = np.zeros((self.cdf_sta_obs_num, 3, self.time_steps))

        "storage for dynamic cdf obstacles"
        if cdf_dyn_obs_params is not None:
            self.cdf_obs_cbf_t = np.zeros((self.cdf_dyn_obs_num, 1, self.time_steps))
            self.cdf_obs_dx_cbf_t = np.zeros((self.cdf_dyn_obs_num, 3, self.time_steps))

        "storage for static sdf obstacles"
        if obs_params is not None:
            self.sdf_obs_cbf_t = np.zeros((self.sdf_sta_obs_num, 1, self.time_steps))
            self.sdf_obs_dx_cbf_t = np.zeros((self.sdf_sta_obs_num, 2, self.time_steps))

        # plot
        self.ani = Render_Animation(
            robot_params,
            cir_obs_params,
            self.step_time,
        )
        self.show_obs = True

    def navigation_destination(self, sdf=None, case_flag=None, add_slack=False):
        """ navigate the robot to its destination """
        t = 0
        process_time = []
        # approach the destination or exceed the maximum time
        dist_to_goal = np.inf

        while dist_to_goal >= self.destination_margin and t - self.time_steps < 0.0:
            if t % 100 == 0:
                print(f't = {t}')

            start_time = time.time()
            if case_flag == 3:
                eef_pos_task = np.array([self.l[0] * np.cos(self.robot_cur_state[0]) +
                                         self.l[1] * np.cos(self.robot_cur_state[0] + self.robot_cur_state[1]),
                                         self.l[0] * np.sin(self.robot_cur_state[0]) +
                                         self.l[1] * np.sin(self.robot_cur_state[0] + self.robot_cur_state[1])])
                dist_to_goal = np.linalg.norm(eef_pos_task - self.robot_target_state[0:2])
                optimal_result = self.cbf_qp.clf_qp(self.robot_cur_state, add_slack=add_slack, case_flag=case_flag)

            elif case_flag == 5:
                cdf.obj_lists = [Circle(center=torch.from_numpy(self.sdf_sta_obs_list[0].state),
                                        radius=self.sdf_sta_obs_list[0].radius, device=device)]
                robot_states = torch.from_numpy(self.robot_cur_state[:2]).to(device).reshape(1, 2)
                distance_input, gradient_input = cdf.inference_c_space_sdf_using_data(robot_states)
                distance_input = distance_input.cpu().detach().numpy()
                gradient_input = gradient_input.cpu().detach().numpy()
                dist_to_goal = distance_input - self.destination_margin
                optimal_result = self.cbf_qp.clf_qp(self.robot_cur_state, add_slack=add_slack, case_flag=case_flag,
                                                    dist=dist_to_goal, grad=gradient_input)

            process_time.append(time.time() - start_time)

            if not optimal_result.feas:
                print('This problem is infeasible, we can not get a feasible solution!')
                break
            else:
                pass

            self.xt[:, t] = np.copy(self.robot_cur_state)
            self.ut[:, t] = optimal_result.u
            self.clft[0, t] = optimal_result.clf

            # update the state of robot
            self.robot_cur_state = self.cbf_qp.robot.next_state(self.robot_cur_state, optimal_result.u, self.step_time)
            t = t + 1

        self.terminal_time = t
        # storage the last state of robot
        self.xt[:, t] = np.copy(self.robot_cur_state)
        self.show_obs = False

        print('Total time: ', self.terminal_time)
        if dist_to_goal <= self.destination_margin:
            print('Robot has arrived its destination!')
        else:
            print('Robot has not arrived its destination!')
        print('Finish the solve of QP with clf!')

        print('Maxinum_time:', max(process_time))
        print('Minimum_time:', min(process_time))
        print('Median_time:', statistics.median(process_time))
        print('Average_time:', statistics.mean(process_time))

    def collision_avoidance(self, rdf=None, case_flag=None, add_clf=True):
        """ solve the collision avoidance between robot and obstacles based on sdf-cbf """
        t = 0
        process_time = []

        # approach the destination or exceed the maximum time
        rob_pos_task_space = np.array([self.l[0] * np.cos(self.robot_cur_state[0]) + self.l[1] * np.cos(
            self.robot_cur_state[0] + self.robot_cur_state[1]),
                                       self.l[0] * np.sin(self.robot_cur_state[0]) + self.l[1] * np.sin(
                                           self.robot_cur_state[0] + self.robot_cur_state[1])])

        gradient_wrt_robot_plot = []

        while (
                np.linalg.norm(rob_pos_task_space - self.robot_target_state[0:2])
                >= self.destination_margin
                and t - self.time_steps < 0.0
        ):
            if t % 100 == 0:
                print(f't = {t}')

            start_time = time.time()

            if cdf is None and sdf is None:
                optimal_result = self.cbf_qp.cbf_clf_qp(self.robot_cur_state, self.cir_obs_states_list, add_clf=add_clf)
            else:
                if self.sdf_flag:
                    distance_input_list = []
                    gradient_input_list = []
                    for i in range(self.sdf_sta_obs_num):
                        sdf.obj_lists = [Circle(center=torch.from_numpy(self.sdf_sta_obs_list[i].state),
                                                radius=self.sdf_sta_obs_list[i].radius, device=device)]
                        robot_states = torch.from_numpy(self.robot_cur_state[:2]).to(device).reshape(1, 2)
                        dist_infer, grad_infer = sdf.inference_sdf_grad(robot_states)
                        dist_infer = dist_infer.cpu().detach().numpy()
                        grad_infer = grad_infer.cpu().detach().numpy()
                        distance_input_list.append(dist_infer)
                        gradient_input_list.append(grad_infer)
                        gradient_wrt_robot_plot.append(grad_infer)
                    optimal_result = self.cbf_qp.cbf_clf_sdf_qp(self.robot_cur_state, distance_input_list,
                                                                gradient_input_list, add_clf=add_clf)

                else:
                    if self.cdf_dyn_obs_num == 0:
                        cdf.obj_lists = None
                        distance_input_list = []
                        gradient_input_list = []
                        for i in range(self.cdf_sta_obs_num):
                            cdf.obj_lists = [Circle(center=torch.from_numpy(self.cdf_sta_obs_list[i].state),
                                                    radius=self.cdf_sta_obs_list[i].radius, device=device)]
                            robot_states = torch.from_numpy(self.robot_cur_state[:2]).to(device).reshape(1, 2)
                            distance_input, gradient_input = cdf.inference_c_space_sdf_using_data(robot_states)
                            distance_input = distance_input.cpu().detach().numpy()
                            gradient_input = gradient_input.cpu().detach().numpy()
                            gradient_input = np.array([gradient_input[0][0], gradient_input[0][1], 0.0]).reshape(1, 3)
                            distance_input = distance_input - self.margin
                            distance_input_list.append(distance_input)
                            gradient_input_list.append(gradient_input)
                        optimal_result = self.cbf_qp.cbf_clf_cdf_qp(self.robot_cur_state, distance_input_list,
                                                                    gradient_input_list, add_clf=add_clf)

                    else:
                        # cdf.obj_lists = [None for i in range(self.cdf_dyn_obs_num)]
                        # print(t)
                        # for i in range(self.cdf_dyn_obs_num):
                        #     cdf.obj_lists[i] = Circle(center=torch.from_numpy(self.cdf_dyn_obs_list[i].state),
                        #                               radius=self.cdf_dyn_obs_list[i].radius, device=device)
                        # robot_states = torch.from_numpy(self.robot_cur_state[:2]).to(device).reshape(1, 2)
                        # distance_input, gradient_input = cdf.inference_c_space_sdf_using_data(robot_states)
                        # ob_distance_input, ob_gradient_input, ob_state = cdf.inference_t_space_sdf_using_data(robot_states)
                        # distance_input = distance_input.cpu().detach().numpy()
                        # gradient_input = gradient_input.cpu().detach().numpy()
                        # ob_distance_input = ob_distance_input.cpu().detach().numpy()
                        # ob_gradient_input = ob_gradient_input.cpu().detach().numpy()
                        # ob_state = ob_state.cpu().detach().numpy()

                        cdf.obj_lists = None
                        distance_input_list = []
                        gradient_input_list = []
                        ob_distance_input_list = []
                        ob_gradient_input_list = []
                        ob_state_list = []
                        for i in range(self.cdf_dyn_obs_num):
                            cdf.obj_lists = [Circle(center=torch.from_numpy(self.cdf_dyn_obs_list[i].state),
                                                    radius=self.cdf_dyn_obs_list[i].radius, device=device)]
                            robot_states = torch.from_numpy(self.robot_cur_state[:2]).to(device).reshape(1, 2)
                            distance_input, gradient_input = cdf.inference_c_space_sdf_using_data(robot_states)
                            ob_distance_input, ob_gradient_input, ob_state = cdf.inference_t_space_sdf_using_data(
                                robot_states)
                            distance_input = distance_input.cpu().detach().numpy()
                            gradient_input = gradient_input.cpu().detach().numpy()
                            distance_input = distance_input - self.margin
                            gradient_input = np.array([gradient_input[0][0], gradient_input[0][1], 0.0]).reshape(1, 3)
                            ob_distance_input = ob_distance_input.cpu().detach().numpy()
                            ob_gradient_input = ob_gradient_input.cpu().detach().numpy()
                            ob_state = ob_state.cpu().detach().numpy()
                            distance_input_list.append(distance_input)
                            gradient_input_list.append(gradient_input)
                            ob_distance_input_list.append(ob_distance_input)
                            ob_gradient_input_list.append(ob_gradient_input)
                            ob_state_list.append(ob_state)

                        self.dyn_obstacle_gradient_filed.append(np.hstack((ob_state_list, ob_gradient_input_list)))

                        optimal_result = self.cbf_qp.cbf_clf_dyn_cdf_qp(self.robot_cur_state, distance_input_list,
                                                                        gradient_input_list, ob_gradient_input_list,
                                                                        ob_state_list, self.cdf_dyn_obs_list,
                                                                        add_clf=add_clf)

                        # optimal_result = self.cbf_qp.cbf_clf_dyn_cdf_qp(self.robot_cur_state, distance_input,
                        #                                                 gradient_input, ob_gradient_input, ob_state,
                        #                                                 self.cdf_dyn_obs_list, add_clf=add_clf)

            process_time.append(time.time() - start_time)

            if not optimal_result.feas:
                print('This problem is infeasible, we can not get a feasible solution!')
                break
            else:
                pass

            self.ut[:, t] = optimal_result.u
            self.clft[0, t] = optimal_result.clf
            self.slackt[0, t] = optimal_result.slack

            # storage and update the state of robot and obstacle
            self.xt[:, t] = np.copy(self.robot_cur_state)
            self.robot_cur_state = self.cbf_qp.robot.next_state(self.robot_cur_state, optimal_result.u, self.step_time)
            rob_pos_task_space = np.array([self.l[0] * np.cos(self.robot_cur_state[0]) + self.l[1] * np.cos(
                self.robot_cur_state[0] + self.robot_cur_state[1]),
                                           self.l[0] * np.sin(self.robot_cur_state[0]) + self.l[1] * np.sin(
                                               self.robot_cur_state[0] + self.robot_cur_state[1])])
            if cdf is None:
                if self.cir_obs_states_list is not None:
                    self.cir_obs_cbf_t[:, t] = optimal_result.cir_cbf_list

                    for i in range(self.cir_obs_num):
                        self.cir_obstacle_state_t[i][:, t] = np.copy(self.cir_obs_states_list[i])
                        self.cir_obs_dx_cbf_t[i][:, t] = (optimal_result.cir_dx_cbf_list[i])[0][0:2]
                        self.cir_obs_dot_cbf_t[i][:, t] = (optimal_result.cir_do_cbf_list[i])[0][0:2]
                        # update the state of dynamic obstacles
                        self.cir_obs_list[i].move_forward(self.step_time)
                    self.cir_obs_states_list = [self.cir_obs_list[i].get_current_state() for i in
                                                range(self.cir_obs_num)]
                if self.sdf_flag:
                    for i in range(self.sdf_sta_obs_num):
                        self.sdf_obs_cbf_t[i][:, t] = optimal_result.sdf_cbf_list[i]
                        self.sdf_obs_dx_cbf_t[i][:, t] = optimal_result.sdf_dx_cbf_list[i]

            else:
                if self.cdf_dyn_obs_num == 0:
                    for i in range(len(distance_input_list)):
                        self.cdf_obs_cbf_t[i][:, t] = optimal_result.cdf_cbf_list[i]
                        self.cdf_obs_dx_cbf_t[i][:, t] = (optimal_result.cdf_dx_cbf_list[i])
                else:
                    for i in range(len(distance_input_list)):
                        self.cdf_obs_cbf_t[i][:, t] = optimal_result.cdf_cbf_list[i]
                        self.cdf_obs_dx_cbf_t[i][:, t] = (optimal_result.cdf_dx_cbf_list[i])

                    for i in range(self.cdf_dyn_obs_num):
                        # update the state of dynamic obstacles
                        self.cdf_dyn_obs_list[i].move_forward(self.step_time)
                    self.cdf_dyn_obs_states_list = [self.cdf_dyn_obs_list[i].get_current_state() for i in
                                                    range(self.cdf_dyn_obs_num)]
                    self.cdf_dyn_obs_center_list.append(np.array(self.cdf_dyn_obs_states_list))

            # update the time
            t = t + 1

        self.terminal_time = t

        # storage the last state of robot and obstacles
        self.xt[:, t] = np.copy(self.robot_cur_state)
        if cdf is None:
            if self.cir_obs_states_list is not None:
                for i in range(self.cir_obs_num):
                    self.cir_obstacle_state_t[i][:, t] = np.copy(self.cir_obs_states_list[i])

        print('Total time: ', self.terminal_time)
        if np.linalg.norm(rob_pos_task_space - self.robot_target_state[0:2]) <= self.destination_margin:
            print('Robot has arrived its destination!')
        else:
            print('Robot has not arrived its destination!')
        print('Finish the solve of QP with clf and cbf!')

        print('Maxinum_time:', max(process_time))
        print('Minimum_time:', min(process_time))
        print('Median_time:', statistics.median(process_time))
        print('Average_time:', statistics.mean(process_time))

    def render(self, i):
        self.ani.render(i, self.xt, self.cir_obstacle_state_t, self.terminal_time, self.show_obs, self.cir_obs_dx_cbf_t
                        , self.cir_obs_dot_cbf_t)

    def render_sdf_static(self, cdf, xt, terminal_time):
        self.ani.render_sdf_ani_static(cdf, xt, terminal_time)

    def render_cdf(self, cdf):
        self.ani.render_cdf(cdf, self.xt, self.cdf_sta_obs_list, self.terminal_time, self.show_obs,
                            self.cdf_obs_dx_cbf_t, show_arrow=True)

    def render_dynamic_cdf(self, cdf, log_circle_center, log_gradient_field):
        self.ani.render_dynamic_cdf(cdf, log_circle_center, log_gradient_field, self.xt, self.terminal_time,
                                    self.show_obs, self.cdf_obs_dx_cbf_t, self.cdf_dyn_obs_num, show_arrow=True,
                                    show_ob_arrow=True)

    def render_manipulator(self, case_flag):
        self.ani.render_manipulator(cdf, self.xt, self.terminal_time, case_flag)

    def render_c_space(self, case_flag):
        self.ani.render_c_space(cdf, self.xt, self.terminal_time, case_flag)

    def render_ani_manipulator(self, cdf, log_circle_center):
        self.ani.render_ani_manipulator(cdf, log_circle_center, self.xt, self.cdf_dyn_obs_num, self.terminal_time)

    def render_sta_ani_manipulator(self, cdf, circle_center):
        self.ani.render_sta_ani_manipulator(cdf, circle_center, self.xt, self.cdf_sta_obs_num, self.terminal_time)

    def render_sta_sdf_manipulator(self, sdf, circle_center, given_joint_angles, terminal_time):
        self.ani.render_sta_sdf_ani_manipulator(sdf, self.xt, circle_center, given_joint_angles, terminal_time)

    def render_dyn_sdf_manipulator(self):
        pass

    def show_cbf(self, i):
        self.ani.show_cbf(i, self.sdf_obs_cbf_t, self.terminal_time)

    def show_cdf_cbf(self, i):
        self.ani.show_cdf_cbf(i, self.cdf_obs_cbf_t, self.terminal_time)

    def show_controls(self):
        self.ani.show_integral_controls(self.ut, self.terminal_time)

    def show_clf(self):
        self.ani.show_clf(self.clft[0], self.terminal_time)

    def show_slack(self):
        self.ani.show_slack(self.slackt[0], self.terminal_time)

    def show_dx_cbf(self, i):
        self.ani.show_dx_cbf(i, self.cir_obs_dx_cbf_t, self.terminal_time)


if __name__ == '__main__':
    """
    sdf yaml files use the traditional way to compute the distance and gradient between robot and objects.
    cdf yaml files use the NN to predict the distance and gradient in configuration space.
    rdf yaml files use the NN to predict the distance and gradient in task space.
    """

    file_names = {
        1: 'static_sdf_cbf_clf.yaml',
        2: 'dynamic_sdf_cbf_clf.yaml',
        3: './eef_static_rdf_clf.yaml',
        4: 'eef_dynamic_rdf_clf.yaml',
        5: './yaml_files/static_rdf_clf.yaml',
        6: 'dynamic_rdf_clf.yaml',
        7: 'static_rdf_clf_cbf.yaml',
        8: 'dynamic_rdf_clf_cbf.yaml',
    }

    case = 5
    file_name = os.path.join(CURRENT_DIR, file_names[case])
    cdf = CDF2D(device)
    test_target = Collision_Avoidance(file_name, case)

    distance_fields = {
        1: 'sdf',
        2: 'rdf',
    }

    if case == 1:
        pass
    elif case == 2:
        pass
    elif case == 3:
        test_target.navigation_destination(case_flag=3)
        test_target.render_manipulator(case_flag=3)

    elif case == 4:
        pass
    elif case == 5:
        test_target.navigation_destination(sdf=cdf, case_flag=5, add_slack=True)
        test_target.render_manipulator(case_flag=5)
        test_target.render_c_space(case_flag=5)
        test_target.show_controls()
        test_target.show_clf()
        test_target.show_slack()
    elif case == 6:
        pass
    elif case == 7:
        test_target.collision_avoidance(rdf=cdf)

    # if case == 1:
    #     "collision avoidance with circle cbf"
    #     # compute the target state in the task space according to the given joint angles
    #     # given_joint_angles = np.array([1.0, 2.0])
    #     # target_state = \
    #     #     cdf.robot.forward_kinematics_all_joints(torch.from_numpy(given_joint_angles).to(device).unsqueeze(0))[
    #     #         0].detach().cpu().numpy()[:, -1]
    #     # print('target_state:', target_state)
    #
    #     # close the file
    #     test_target.render_manipulator()
    #
    #     test_target.render_sdf_static(cdf, test_target.xt, test_target.terminal_time)
    #     test_target.render_sta_sdf_manipulator(cdf, test_target.sdf_sta_obs_list, given_joint_angles,
    #                                            test_target.terminal_time)
    #     test_target.show_cbf(0)
    #     test_target.show_controls()
    #     test_target.show_clf()
    #     test_target.show_slack()
    # test_target.show_dx_cbf(0)

    # elif case == 2:
    #     "collision avoidance with dynamic circle cbf"
    #     test_target.collision_avoidance()
    #     test_target.render(0)
    #     test_target.show_controls()
    #     test_target.show_clf()
    #     test_target.show_slack()
    #     test_target.show_cbf(0)
    #     test_target.show_dx_cbf(0)
    #
    # elif case == 3:
    #     "collision avoidance with static cdf cbf"
    #     test_target.collision_avoidance(cdf=cdf)
    #     test_target.render_cdf(cdf)
    #     test_target.render_manipulator()
    #     test_target.render_sta_ani_manipulator(cdf, test_target.cdf_sta_obs_list)
    #     test_target.show_clf()
    #     test_target.show_cdf_cbf(0)
    #     # test_target.show_cdf_cbf(1)  # show the cbf of the second obstacle
    #     test_target.show_controls()
    #     test_target.show_slack()
    #
    # elif case == 4:
    #     "collision avoidance with dynamic cdf cbf"
    #     test_target.collision_avoidance(cdf=cdf)
    #     test_target.render_dynamic_cdf(cdf, test_target.cdf_dyn_obs_center_list,
    #                                    test_target.dyn_obstacle_gradient_filed)
    #     test_target.render_ani_manipulator(cdf, test_target.cdf_dyn_obs_center_list)
    #     test_target.show_clf()
    #     test_target.show_cdf_cbf(0)
    #     # test_target.show_cdf_cbf(1)  # show the cbf of the second obstacle
    #     test_target.show_controls()
    #     test_target.show_slack()
