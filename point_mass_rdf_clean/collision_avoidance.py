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
        obj_params = None

        self.obs_gradient_filed = []
        self.dyn_obs_state_list = []
        self.dyn_obs_center_list = []

        if case_flag == 1:
            pass
        elif case_flag == 2:
            pass
        elif case_flag == 3:
            obs_params = config.get('eef_static_sdf_clf_list')
        elif case_flag == 4:
            pass
        elif case_flag == 5:
            obj_params = config.get('obj_list')
        elif case_flag == 6:
            pass
        elif case_flag == 7:
            obs_params = config.get('obs_list')
            obj_params = config.get('obj_list')
        elif case_flag == 8:
            obs_params = config.get('obs_list')
            obj_params = config.get('obj_list')

        if obj_params is not None:
            self.obj_num = len(obj_params['obj_states'])
            self.obj_list = [None for i in range(self.obj_num)]
            for i in range(self.obj_num):
                self.obj_list[i] = obs.Cdf_Obs(
                    index=i,
                    radius=obj_params['obj_radii'][i],
                    center=obj_params['obj_states'][i],
                    vel=obj_params['obj_vel'][i],
                    mode=obj_params['modes'][i],
                )
        if obs_params is not None:
            self.obs_num = len(obs_params['obs_states'])
            self.obs_list = [None for i in range(self.obs_num)]
            for i in range(self.obs_num):
                self.obs_list[i] = obs.Sdf_Obs(
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

        "storage for static sdf obstacles"
        if obs_params is not None:
            self.sdf_obs_cbf_t = np.zeros((self.obs_num, 1, self.time_steps))
            self.sdf_obs_dx_cbf_t = np.zeros((self.obs_num, 2, self.time_steps))

        # plot
        self.ani = Render_Animation(
            robot_params,
            self.step_time,
        )
        self.show_obs = True

    def navigation_destination(self, distance_field=None, cdf=None, case_flag=None, add_slack=False):
        """ navigate the robot to its destination """
        t = 0
        process_time = []
        # approach the destination or exceed the maximum time
        dist_to_goal = np.inf
        distance_input = None
        gradient_input = None

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
                cdf.obj_lists = [Circle(center=torch.from_numpy(self.obj_list[0].state),
                                        radius=self.obj_list[0].radius, device=device)]
                robot_states = torch.from_numpy(self.robot_cur_state[:2]).to(device).reshape(1, 2)
                if distance_field == 'sdf':
                    distance_input, gradient_input = cdf.inference_sdf_grad(robot_states)
                elif distance_field == 'cdf':
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

    def collision_avoidance(self, distance_field, CDF, reach_mode=None, case_flag=None,
                            add_clf=True):
        t = 0
        process_time = []
        dist_to_goal = np.inf
        distance_input = None
        gradient_input = None
        optimal_result = None

        if reach_mode == "eef":
            rob_pos_task_space = np.array([self.l[0] * np.cos(self.robot_cur_state[0]) + self.l[1] * np.cos(
                self.robot_cur_state[0] + self.robot_cur_state[1]),
                                           self.l[0] * np.sin(self.robot_cur_state[0]) + self.l[1] * np.sin(
                                               self.robot_cur_state[0] + self.robot_cur_state[1])])

        elif reach_mode == "point_to_point":
            dist_to_goal = np.linalg.norm(self.robot_cur_state - self.robot_target_state)

        gradient_wrt_robot_plot = []
        u_last = 0.0
        while dist_to_goal >= self.destination_margin and (t - self.time_steps) < 0.0:
            if t % 100 == 0:
                print(f't = {t}')

            start_time = time.time()
            CDF.obj_lists = None

            distance_input_list = []
            gradient_input_list = []
            obs_distance_input_list = []
            obs_gradient_input_list = []
            obs_state_list = []

            if case_flag == 7:
                for i in range(self.obs_num):
                    CDF.obj_lists = [Circle(center=torch.from_numpy(self.obs_list[i].state),
                                            radius=self.obs_list[i].radius, device=device)]
                    robot_states = torch.from_numpy(self.robot_cur_state[:2]).to(device).reshape(1, 2)
                    if distance_field == 'sdf':
                        distance_input, gradient_input = CDF.inference_sdf_grad(robot_states)
                    elif distance_field == 'cdf':
                        distance_input, gradient_input = CDF.inference_c_space_sdf_using_data(robot_states)

                    distance_input = distance_input.cpu().detach().numpy()
                    gradient_input = gradient_input.cpu().detach().numpy()
                    distance_input = distance_input - self.margin
                    distance_input_list.append(distance_input)
                    gradient_input_list.append(gradient_input)

                optimal_result = self.cbf_qp.cbf_clf_qp(self.robot_cur_state, distance_input_list, gradient_input_list,
                                                        case_flag)

            if case_flag == 8:
                cdf.obj_lists = [None for i in range(self.obs_num)]
                robot_states = torch.from_numpy(self.robot_cur_state[:2]).to(device).reshape(1, 2)

                for i in range(self.obs_num):
                    cdf.obj_lists = [Circle(center=torch.from_numpy(self.obs_list[i].state),
                                            radius=self.obs_list[i].radius, device=device)]
                    if distance_field == 'sdf':
                        distance_input, gradient_input = cdf.inference_sdf_grad(robot_states)
                    elif distance_field == 'cdf':
                        distance_input, gradient_input = cdf.inference_c_space_sdf_using_data(robot_states)
                        obs_distance_input, obs_gradient_input, obs_state = cdf.inference_t_space_sdf_using_data(
                            robot_states)
                        obs_distance_input = obs_distance_input.cpu().detach().numpy()
                        obs_gradient_input = obs_gradient_input.cpu().detach().numpy()
                        obs_state = obs_state.cpu().detach().numpy()
                        obs_distance_input_list.append(obs_distance_input)
                        obs_gradient_input_list.append(obs_gradient_input)
                        obs_state_list.append(obs_state)

                    distance_input = distance_input.cpu().detach().numpy()
                    gradient_input = gradient_input.cpu().detach().numpy()

                    distance_input = distance_input - self.margin
                    distance_input_list.append(distance_input)
                    gradient_input_list.append(gradient_input)

                self.obs_gradient_filed.append(np.hstack((obs_state_list, obs_gradient_input_list)))

                optimal_result = self.cbf_qp.cbf_clf_qp(self.robot_cur_state, distance_input_list, gradient_input_list,
                                                        case_flag, obs_grad_list=obs_gradient_input_list,
                                                        obs_pos_list=obs_state_list, obs_list=self.obs_list,
                                                        u_last=u_last)

            process_time.append(time.time() - start_time)

            if not optimal_result.feas:
                print('This problem is infeasible, we can not get a feasible solution!')
                break
            else:
                pass

            self.ut[:, t] = optimal_result.u
            self.clft[0, t] = optimal_result.clf
            self.slackt[0, t] = optimal_result.slack
            u_last = np.copy(optimal_result.u)

            # storage and update the state of robot and obstacle
            self.xt[:, t] = np.copy(self.robot_cur_state)
            self.robot_cur_state = self.cbf_qp.robot.next_state(self.robot_cur_state, optimal_result.u, self.step_time)

            dist_to_goal = np.linalg.norm(self.robot_cur_state[0:2] - self.robot_target_state[0:2])
            for i in range(self.obs_num):
                self.sdf_obs_cbf_t[i][:, t] = optimal_result.sdf_cbf_list[i]
                self.sdf_obs_dx_cbf_t[i][:, t] = optimal_result.sdf_dx_cbf_list[i]

            self.dyn_obs_state_list = [self.obs_list[i].get_current_state() for i in range(self.obs_num)]
            self.dyn_obs_center_list.append(np.array(self.dyn_obs_state_list))
            for i in range(self.obs_num):
                self.obs_list[i].move_forward(self.step_time)

            # update the time
            t = t + 1

        self.terminal_time = t

        # storage the last state of robot and obstacles
        self.xt[:, t] = np.copy(self.robot_cur_state)
        print('Total time: ', self.terminal_time)
        if np.linalg.norm(dist_to_goal) <= self.destination_margin:
            print('Robot has arrived its destination!')
        else:
            print('Robot has not arrived its destination!')
        print('Finish the solve of QP with clf and cbf!')

        print('Maxinum_time:', max(process_time))
        print('Minimum_time:', min(process_time))
        print('Median_time:', statistics.median(process_time))
        print('Average_time:', statistics.mean(process_time))

    def render_manipulator(self, case_flag):
        self.ani.render_manipulator(cdf, self.xt, self.terminal_time, case_flag)

    def render_c_space(self, distance_field, case_flag):
        self.ani.render_c_space(distance_field, cdf, self.xt, self.terminal_time, case_flag)

    def render_ani_t_space_manipulator(self, distance_field, reach_mode, case_flag, obs_info=None, obs_list=None,
                                       save_gif=False, save_path=None):
        self.ani.render_ani_t_space_manipulator(distance_field, cdf, self.xt, self.terminal_time, reach_mode, case_flag,
                                                obs_info=obs_info, obs_list=obs_list, save_gif=save_gif,
                                                save_path=save_path)

    def render_ani_c_space(self, distance_field, case_flag, mode='clf', obs_info=None, obs_list=None,
                           obs_grad_field=None, robo_grad_field=None, save_gif=False, save_path=None):
        self.ani.render_ani_c_space(distance_field, cdf, self.xt, self.terminal_time, case_flag, mode,
                                    obs_info=obs_info, obs_list=obs_list, obs_grad_field=obs_grad_field,
                                    robo_grad_field=robo_grad_field, save_gif=save_gif,
                                    save_path=save_path)

    def show_cbf(self, save_result=False, save_path=None):
        self.ani.show_cbf(self.sdf_obs_cbf_t, self.terminal_time, save_result=save_result,
                          save_path=save_path)

    def show_controls(self, save_result=False, save_path=None):
        self.ani.show_integral_controls(self.ut, self.terminal_time, save_result=save_result,
                                        save_path=save_path)

    def show_clf(self, save_result=False, save_path=None):
        self.ani.show_clf(self.clft[0], self.terminal_time, save_result=save_result,
                          save_path=save_path)

    def show_slack(self, save_result=False, save_path=None):
        self.ani.show_slack(self.slackt[0], self.terminal_time, save_result=save_result,
                            save_path=save_path)

    def show_dx_cbf(self, i, save_result=False, save_path=None):
        self.ani.show_dx_cbf(i, self.cir_obs_dx_cbf_t, self.terminal_time, save_result=save_result,
                             save_path=save_path)


if __name__ == '__main__':
    """
    sdf yaml files use the traditional way to compute the distance and gradient between robot and objects.
    cdf yaml files use the NN to predict the distance and gradient in configuration space.
    rdf yaml files use the NN to predict the distance and gradient in task space.
    """

    file_names = {
        1: '01_static_sdf_cbf_clf.yaml',  # point mass robot
        2: '02_dynamic_sdf_cbf_clf.yaml',  # point mass robot
        3: '03_eef_static_sdf_clf.yaml',  # manipulator and its end-effector
        4: '04_eef_dynamic_sdf_clf.yaml',  # manipulator and its end-effector
        5: '05_static_clf.yaml',  # manipulator and whole body
        6: '06_dynamic_clf.yaml',  # todo: need to implement other cases first, then implement this case
        7: '07_static_clf_cbf.yaml',
        8: '08_dynamic_clf_cbf.yaml',
    }

    case = 8
    file_name = os.path.join(CURRENT_DIR, './yaml_files/', file_names[case])
    cdf = CDF2D(device)
    test_target = Collision_Avoidance(file_name, case)

    distance_fields = {
        1: 'sdf',
        2: 'cdf',
    }

    DF = distance_fields[2]

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
        test_target.navigation_destination(distance_field=DF, cdf=cdf, case_flag=5)
        test_target.render_manipulator(case_flag=5)
        test_target.render_c_space(distance_field=DF, case_flag=5)
        test_target.render_ani_t_space_manipulator(DF, cdf, case_flag=5)  # todo: add mode
        test_target.render_ani_c_space(DF, case_flag=5, mode='clf', save_gif=False)
        test_target.show_controls()
        test_target.show_clf()
        test_target.show_slack()
        # todo: add output of other metrics, such as the length of the solution.
        #
    elif case == 6:
        pass
    elif case == 7:
        reachModeList = {
            1: "eef",  # the goal (x,y) in task space
            2: "whole_body",  # the goal depends on object's position and shape
            3: "point_to_point",  # specify the goal in joint space
        }
        reachMode = reachModeList[3]
        save_files = False
        result_dir = None
        if save_files:
            date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            result_dir = os.path.join(CURRENT_DIR, f'./results/{case}/{date}')
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
        test_target.collision_avoidance(DF, cdf, reach_mode=reachMode, case_flag=case)
        test_target.render_manipulator(case_flag=7)
        test_target.render_c_space(distance_field=DF, case_flag=7)
        test_target.render_ani_t_space_manipulator(DF, reachMode, case_flag=case, obs_list=test_target.obs_list)
        test_target.render_ani_c_space(DF, case_flag=7, mode='clf_cbf', obs_list=test_target.obs_list, save_gif=False)
        test_target.show_cbf()
        test_target.show_clf()
        test_target.show_controls()
        test_target.show_slack()
    elif case == 8:
        reachModeList = {
            1: "eef",  # the goal (x,y) in task space
            2: "whole_body",  # the goal depends on object's position and shape
            3: "point_to_point",  # specify the goal in joint space
        }
        reachMode = reachModeList[3]
        save_files = True
        result_dir = None
        if save_files:
            date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            result_dir = os.path.join(CURRENT_DIR, f'./results/{case}/{date}')
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

        test_target.collision_avoidance(DF, cdf, reach_mode=reachMode, case_flag=case)
        test_target.render_ani_c_space(DF, case_flag=case, mode='clf_cbf', obs_info=test_target.dyn_obs_center_list,
                                       obs_list=test_target.obs_list, obs_grad_field=test_target.obs_gradient_filed,
                                       robo_grad_field=test_target.sdf_obs_dx_cbf_t, save_gif=save_files,
                                       save_path=result_dir)
        test_target.render_ani_t_space_manipulator(DF, reachMode, case_flag=case,
                                                   obs_info=test_target.dyn_obs_center_list,
                                                   obs_list=test_target.obs_list,
                                                   save_gif=save_files, save_path=result_dir)
        test_target.show_cbf(save_result=save_files, save_path=result_dir)
        test_target.show_clf(save_result=save_files, save_path=result_dir)
        test_target.show_controls(save_result=save_files, save_path=result_dir)
        test_target.show_slack(save_result=save_files, save_path=result_dir)
