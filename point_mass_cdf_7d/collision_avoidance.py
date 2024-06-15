import numpy as np
from integral_sdf_qp import Integral_Sdf_Cbf_Clf
import time
import yaml
import obs
import statistics
import os
import torch
import trimesh
from cdf import CDF2D
from primitives2D_torch import Circle
from render_show import Render_Animation
from mlp import MLPRegression
from train_nn_7d import CDF

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
print("CURRENT_DIR:", CURRENT_DIR)


class Collision_Avoidance:
    def __init__(self, file_name) -> None:
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

        "init obstacle, if no data, return None"
        cir_obs_params = config.get('cir_obstacle_list')
        cdf_sta_obs_params = config.get('cdf_sta_obstacle_list')
        cdf_dyn_obs_params = config.get('cdf_dyn_obstacle_list')
        self.cir_obs_states_list = None
        self.cdf_dyn_obs_states_list = None
        self.cdf_dyn_obs_num = 0
        self.cdf_dyn_obs_center_list = []

        if cir_obs_params is not None:
            self.cir_obs_num = len(cir_obs_params['obs_states'])
            self.cir_obs_list = [None for i in range(self.cir_obs_num)]
            for i in range(self.cir_obs_num):
                self.cir_obs_list[i] = obs.Circle_Obs(
                    index=i,
                    radius=cir_obs_params['obs_radii'][i],
                    center=cir_obs_params['obs_states'][i],
                    vel=cir_obs_params['obs_vel'][i],
                    mode=cir_obs_params['modes'][i],
                )

            # get cir_obstacles' center position and velocity as well as radius
            self.cir_obs_init_states_list = [
                self.cir_obs_list[i].get_current_state()
                for i in range(self.cir_obs_num)
            ]

            self.cir_obs_vel = [
                self.cir_obs_list[i].get_current_vel()
                for i in range(self.cir_obs_num)
            ]

            self.cir_obs_states_list = np.copy(self.cir_obs_init_states_list)

        if cdf_sta_obs_params is not None:
            self.cdf_sta_obs_num = cdf_sta_obs_params['obs_num']
            self.cdf_sta_obs_list = [None for i in range(self.cdf_sta_obs_num)]
            # for i in range(self.cdf_sta_obs_num):
            #     self.cdf_sta_obs_list[i] = obs.Cdf_Obs(
            #         index=i,
            #         radius=cdf_sta_obs_params['obs_radii'][i],
            #         center=cdf_sta_obs_params['obs_states'][i],
            #         vel=cdf_sta_obs_params['obs_vel'][i],
            #         mode=cdf_sta_obs_params['modes'][i],
            #     )

        if cdf_dyn_obs_params is not None:
            self.dyn_obstacle_gradient_filed = []
            self.cdf_dyn_obs_num = len(cdf_dyn_obs_params['obs_states'])
            self.cdf_dyn_obs_list = [None for i in range(self.cdf_dyn_obs_num)]
            for i in range(self.cdf_dyn_obs_num):
                self.cdf_dyn_obs_list[i] = obs.Cdf_Obs(
                    index=i,
                    radius=cdf_dyn_obs_params['obs_radii'][i],
                    center=cdf_dyn_obs_params['obs_states'][i],
                    vel=cdf_dyn_obs_params['obs_vel'][i],
                    mode=cdf_dyn_obs_params['modes'][i],
                )

        # controller
        self.T = controller_params['Tmax']
        self.step_time = controller_params['step_time']
        self.time_steps = int(np.ceil(self.T / self.step_time))
        self.terminal_time = self.time_steps

        # storage
        self.xt = np.zeros((7, self.time_steps + 1))
        self.ut = np.zeros((7, self.time_steps))
        self.clft = np.zeros((1, self.time_steps))
        self.slackt = np.zeros((1, self.time_steps))

        "storage for circle obstacles"
        if cir_obs_params is not None:
            self.cir_obstacle_state_t = None
            self.cir_obs_cbf_t = None
            self.cir_obs_dx_cbf_t = None
            self.cir_obstacle_state_t = np.zeros((self.cir_obs_num, 8, self.time_steps + 1))
            self.cir_obs_cbf_t = np.zeros((self.cir_obs_num, self.time_steps))
            self.cir_obs_dx_cbf_t = np.zeros((self.cir_obs_num, 2, self.time_steps))
            self.cir_obs_dot_cbf_t = np.zeros((self.cir_obs_num, 2, self.time_steps))

        "storage for static cdf obstacles"
        if cdf_sta_obs_params is not None:
            self.cdf_obs_cbf_t = np.zeros((self.cdf_sta_obs_num, 1, self.time_steps))
            self.cdf_obs_dx_cbf_t = np.zeros((self.cdf_sta_obs_num, 7, self.time_steps))

        "storage for dynamic cdf obstacles"
        if cdf_dyn_obs_params is not None:
            self.cdf_obs_cbf_t = np.zeros((self.cdf_dyn_obs_num, 1, self.time_steps))
            self.cdf_obs_dx_cbf_t = np.zeros((self.cdf_dyn_obs_num, 7, self.time_steps))
        # plot
        self.ani = Render_Animation(
            robot_params,
            cir_obs_params,
            self.step_time,
        )
        self.show_obs = True

    def navigation_destination(self, add_slack=False):
        """ navigate the robot to its destination """
        t = 0
        process_time = []
        # approach the destination or exceed the maximum time
        while (
                np.linalg.norm(self.robot_cur_state[0:2] - self.robot_target_state[0:2])
                >= self.destination_margin
                and t - self.time_steps < 0.0
        ):
            if t % 100 == 0:
                print(f't = {t}')

            start_time = time.time()
            optimal_result = self.cbf_qp.clf_qp(self.robot_cur_state, add_slack=add_slack)
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
        if np.linalg.norm(self.robot_cur_state[0:2] - self.robot_target_state[0:2]) <= self.destination_margin:
            print('Robot has arrived its destination!')
        else:
            print('Robot has not arrived its destination!')
        print('Finish the solve of QP with clf!')

        print('Maxinum_time:', max(process_time))
        print('Minimum_time:', min(process_time))
        print('Median_time:', statistics.median(process_time))
        print('Average_time:', statistics.mean(process_time))

    def collision_avoidance(self, cdf=None, add_clf=True):
        """ solve the collision avoidance between robot and obstacles based on sdf-cbf """
        t = 0
        process_time = []
        distance_input = None
        gradient_input = None

        wall_size = torch.tensor([0.5, 0.5]).to(device)
        wall_center = torch.tensor([0.5, 0.0, 0.2]).to(device)
        wall_rot = torch.tensor([[1.0, 0.0, 0.0],
                                 [0.0, 0.0, -1.0],
                                 [0.0, 1.0, 0.0], ]).to(device)
        p = wall(wall_size, wall_center, wall_rot)
        # p = p[0].reshape(-1, 3) + torch.tensor([0.25, 0.0, 0.35]).to(device)
        # put obstacles here
        # ring_radius = 0.3
        # ring_center = torch.tensor([0.4, 0.0, 0.45]).to(device)
        # ring_rot = torch.tensor([[0.0, 0.0, -1.0],
        #                          [0.0, 1.0, 0.0],
        #                          [1.0, 0.0, 0.0], ]).to(device)
        # p = ring(ring_radius, ring_center, ring_rot)

        self.cdf_sta_obs_num = p.shape[0]

        # approach the destination or exceed the maximum time
        while (
                np.linalg.norm(self.robot_cur_state[0:7] - self.robot_target_state[0:7])
                >= self.destination_margin
                and t - self.time_steps < 0.0
        ):
            if t % 100 == 0:
                print(f't = {t}')

            start_time = time.time()

            if cdf is None:
                optimal_result = self.cbf_qp.cbf_clf_qp(self.robot_cur_state, self.cir_obs_states_list,
                                                        self.cir_obs_vel, add_clf=add_clf)
            else:
                if self.cdf_dyn_obs_num == 0:
                    cdf.obj_lists = None
                    distance_input_list = []
                    gradient_input_list = []
                    for i in range(p.shape[0]):
                        # for i in range(1):
                        # cdf.obj_lists = [Circle(center=torch.from_numpy(self.cdf_sta_obs_list[i].state),
                        #                         radius=self.cdf_sta_obs_list[i].radius, device=device)]
                        robot_states = torch.from_numpy(self.robot_cur_state[:7]).to(device).float().reshape(1, 7)
                        robot_states.requires_grad = True
                        distance_input, gradient_input = cdf.inference_d_wrt_q(p[i].reshape(1, 3), robot_states, model,
                                                                               return_grad=True)
                        # distance_input, gradient_input = cdf.inference_d_wrt_q(p, robot_states, model,
                        #                                                        return_grad=True)
                        # distance_input, gradient_input = cdf.inference_c_space_sdf_using_data(robot_states)
                        distance_input = distance_input.cpu().detach().numpy()
                        gradient_input = gradient_input.cpu().detach().numpy()
                        # gradient_input = np.hstack((gradient_input, np.array([[0.0]])))
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
                        robot_states = torch.from_numpy(self.robot_cur_state[:2]).to(device).reshape(1, 7)
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
            self.obs_7d = p

            # storage and update the state of robot and obstacle
            self.xt[:, t] = np.copy(self.robot_cur_state)
            self.robot_cur_state = self.cbf_qp.robot.next_state(self.robot_cur_state, optimal_result.u, self.step_time)

            if cdf is None:
                if self.cir_obs_states_list is not None:
                    self.cir_obs_cbf_t[:, t] = optimal_result.cir_cbf_list

                    for i in range(self.cir_obs_num):
                        self.cir_obstacle_state_t[i][:, t] = np.copy(self.cir_obs_states_list[i])
                        self.cir_obs_dx_cbf_t[i][:, t] = (optimal_result.cir_dx_cbf_list[i])[0][0:2]
                        self.cir_obs_dot_cbf_t[i][:, t] = (optimal_result.cir_do_cbf_list[i])[0][0:2]
                        # update the state of dynamic obstacles
                        self.cir_obs_list[i].move_forward(self.step_time)  # todo: 为什么不会报错？
                    self.cir_obs_states_list = [self.cir_obs_list[i].get_current_state() for i in
                                                range(self.cir_obs_num)]
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
        if np.linalg.norm(self.robot_cur_state[0:7] - self.robot_target_state[0:7]) <= self.destination_margin:
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

    def render_cdf(self, cdf):
        self.ani.render_cdf(cdf, self.xt, self.cdf_sta_obs_list, self.terminal_time, self.show_obs,
                            self.cdf_obs_dx_cbf_t, show_arrow=True)

    def render_dynamic_cdf(self, cdf, log_circle_center, log_gradient_field):
        self.ani.render_dynamic_cdf(cdf, log_circle_center, log_gradient_field, self.xt, self.terminal_time,
                                    self.show_obs, self.cdf_obs_dx_cbf_t, self.cdf_dyn_obs_num, show_arrow=True,
                                    show_ob_arrow=True)

    def render_manipulator(self):
        self.ani.render_manipulator(cdf, self.xt, self.terminal_time)

    def render_ani_manipulator(self, cdf, log_circle_center):
        self.ani.render_ani_manipulator(cdf, log_circle_center, self.xt, self.cdf_dyn_obs_num, self.terminal_time)

    def show_cbf(self, i):
        self.ani.show_cbf(i, self.cir_obs_cbf_t, self.terminal_time)

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


def ring(radius, center, rot):
    # center: (3,)
    # size: (3,)
    # return: (N,3)
    # the gap of the linspace is 0.1
    theta = torch.arange(0, 2 * PI, 0.05).to(device)
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    z = torch.zeros_like(x).to(device)
    points = torch.stack([x, y, z], dim=-1)
    points = torch.matmul(points, rot.transpose(0, 1)) + center
    return points


def mannully_observe_q(q):
    pose = torch.eye(4).unsqueeze(0).to(device).float()
    robot_mesh = []
    for _q in q:
        rm = cdf.panda.get_forward_robot_mesh(pose, _q.unsqueeze(0))[0]
        mesh = np.sum(rm)
        mesh.visual.face_colors = [0, 255, 0, 100]
        robot_mesh.append(mesh)
    return q, robot_mesh


def wall(size, center, rot):
    # center: (3,)
    # size: (3,)
    # return: (N,3)
    # the gap of the linspace is 0.1
    x = torch.arange(-size[0] / 2, size[0] / 2, 0.1).to(device)
    y = torch.arange(-size[1] / 2, size[1] / 2, 0.1).to(device)
    x, y = torch.meshgrid(x, y)
    x, y = x.reshape(-1), y.reshape(-1)
    z = torch.zeros_like(x).to(device)
    points = torch.stack([x, y, z], dim=-1)
    points = torch.matmul(points, rot.transpose(0, 1)) + center
    return points


if __name__ == '__main__':
    file_names = {
        1: 'static_setting.yaml',
        2: 'dynamic_setting.yaml',
        3: 'static_cdf_setting.yaml',
        4: 'dynamic_cdf_setting.yaml'

    }
    PI = 3.14

    case = 3
    file_name = os.path.join(CURRENT_DIR, file_names[case])
    cdf = CDF(device)
    model = MLPRegression(input_dims=10, output_dims=1, mlp_layers=[1024, 512, 256, 128, 128], skips=[],
                          act_fn=torch.nn.ReLU, nerf=True)
    model.load_state_dict(torch.load(os.path.join(CURRENT_DIR, 'model_dict_tension_2.pt'))[49900])
    model.to(device)

    #
    # x0_7d_torch = torch.tensor([-0.41302193, -0.94202107, -0.32044236, -2.47843634, -0.16166646,
    #                             1.74380392, 0.75803596]).to(device).reshape(1, 7)
    # x0_7d_torch.requires_grad = True
    # # distance_input, gradient_input = cdf.inference_d_wrt_q(p, x0_7d_torch, model, return_grad=True)
    # # print('distance_input:', distance_input)
    # q, robot_mesh = mannully_observe_q(x0_7d_torch)
    # scene.add_geometry(robot_mesh[0])
    # scene.show()

    test_target = Collision_Avoidance(file_name)

    "hard code to visualize the robot and obstacles"

    # if case == 1:
    #     "collision avoidance with circle cbf"
    #     test_target.collision_avoidance()
    #     # test_target.render(0)
    #     # test_target.show_controls()
    #     test_target.show_clf()
    #     # test_target.show_slack()
    #     test_target.show_cbf(0)
    #     # test_target.show_dx_cbf(0)
    #
    # # elif case == 2:
    # #     "collision avoidance with dynamic circle cbf"
    # #     test_target.collision_avoidance()
    # #     test_target.render(0)
    # #     test_target.show_controls()
    # #     test_target.show_clf()
    # #     test_target.show_slack()
    # #     test_target.show_cbf(0)
    # #     test_target.show_dx_cbf(0)
    # #
    # elif case == 3:
    #     "collision avoidance with static cdf cbf"
    test_target.collision_avoidance(cdf=cdf)
    test_target.show_cdf_cbf(0)
    test_target.show_controls()
    test_target.show_clf()
    test_target.show_slack()
    scene = trimesh.Scene()

    "Visualize the obstacles"
    robot_q = test_target.xt.T
    for p0 in test_target.obs_7d.data.cpu().numpy():
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
        sphere.visual.face_colors = [255, 0, 0, 100]
        sphere.apply_translation(p0)
        scene.add_geometry(sphere)

    "Visualize the robot trajectory"
    q_init, robot_mesh_init = mannully_observe_q(
        torch.from_numpy(test_target.robot_init_state.reshape(1, 7)).to(device).float())
    robot_mesh_init[0].visual.face_colors = [255, 0, 0, 100]
    scene.add_geometry(robot_mesh_init[0])

    q_target, robot_mesh_target = mannully_observe_q(
        torch.from_numpy(test_target.robot_target_state.reshape(1, 7)).to(device).float())
    robot_mesh_target[0].visual.face_colors = [0, 191, 255, 100]
    scene.add_geometry(robot_mesh_target[0])

    robot_q = robot_q[::2, :]  # downsample the trajectory
    q, robot_mesh_final = mannully_observe_q(torch.from_numpy(robot_q).to(device).float())
    robot_mesh_final[0].visual.face_colors = [0, 0, 255, 250]

    for _q, m in zip(q, robot_mesh_final):
        m.visual.face_colors = [0, 255, 0, 100]
        scene.add_geometry(m)
    scene.show()
