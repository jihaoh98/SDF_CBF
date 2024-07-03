import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import numpy as np
import torch
import os
from primitives2D_torch import Circle

CUR_PATH = os.path.dirname(os.path.realpath(__file__))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Render_Animation:
    def __init__(self, robot_params, dt) -> None:
        """ init the render animation """
        self.dt = dt

        # robot
        self.robot_init_state = np.array(robot_params['initial_state'])
        self.robot_target_state = np.array(robot_params['target_state'])
        self.robot_radius = robot_params['radius']
        self.umax = robot_params['u_max']
        self.umin = robot_params['u_min']

        # # obstacle
        # if cir_obs_params is not None:
        #     self.cir_obs_num = len(cir_obs_params['obs_states'])
        #     self.cir_obs = [None for i in range(self.cir_obs_num)]

        # storage the past states of robots and different shaped obs
        self.xt = None
        self.cir_obs_list_t = None

        # plot
        # self.fig, self.ax = plt.subplots()
        self.ax, self.fig = None, None

        # start and end state of robot
        self.start_body = None
        self.start_arrow = None
        self.end_body = None

        # current state
        self.robot_body = None
        self.robot_arrow = None
        self.docbf_arrow = None

        self.gradientField = []
        self.docbfield = None

        self.show_obs = True
        self.show_arrow = False
        self.show_ob_arrow = False
        self.sample_num = 40

        self.color_palette = ['red', 'black', 'yellow']

        # settings of Times New Roman
        # set the text in Times New Roman
        config = {
            "font.family": 'serif',
            "font.size": 12,
            "font.serif": ['Times New Roman'],
            "mathtext.fontset": 'stix',
        }
        plt.rcParams.update(config)

        # label_font and legend font
        self.label_font = {'family': 'Times New Roman',
                           'weight': 'normal',
                           'size': 16,
                           }
        self.legend_font = {"family": "Times New Roman", "weight": "normal", "size": 10}

    def render_ani_c_space(self, distance_field, cdf, xt, terminal_time, case_flag=None, mode=None, obs_info=None,
                           obs_list=None, obs_grad_field=None, robo_grad_field=None, save_gif=False,
                           save_path=None, show_arrow=True):
        # line is used to update the trajectory
        self.fig, self.ax = plt.subplots()
        line, = self.ax.plot([], [], color='yellow', linestyle='-', linewidth=2)
        num_frames = terminal_time
        self.xt = xt

        num_obs = len(obs_list)

        if mode == "clf":
            # plot the start point, `l1` is the handle for creating a legend
            l1, = self.ax.plot(xt[0, 0], xt[1, 0], 'g*', label='start', markersize=10)
            num_obs = 0
            if distance_field == "sdf":
                cdf.plot_sdf()
            elif distance_field == "cdf":
                d_grad, grad_plot = cdf.inference_c_space_sdf_using_data(cdf.Q_sets)
                contour, contourf, ct_zero, hatch_handle = cdf.plot_cdf_ax(d_grad.detach().cpu().numpy(), self.ax)
        elif mode == "clf_cbf":
            l1, = self.ax.plot(xt[0, 0], xt[1, 0], 'g*', label='start', markersize=10)
            if distance_field == "sdf":
                cdf.plot_sdf()
            elif distance_field == "cdf":
                if case_flag == 7:
                    d_grad, grad_plot = cdf.inference_c_space_sdf_using_data(cdf.Q_sets)
                    contour, contourf, ct_zero, hatch_handle = cdf.plot_cdf_ax(d_grad.detach().cpu().numpy(), self.ax)
                    # plot the goal point
                    l2, = self.ax.plot(self.robot_target_state[0], self.robot_target_state[1], 'r*', label='goal',
                                       markersize=10)
                    self.ax.legend(handles=[l1, l2], loc='upper center', ncol=2)
                if case_flag == 8:
                    l2, = self.ax.plot(self.robot_target_state[0], self.robot_target_state[1], 'r*', label='goal',
                                       markersize=10)
                    if show_arrow:
                        gradientField = np.zeros((num_obs, 2, terminal_time))
                        for i in range(num_obs):
                            gradientField[i] = robo_grad_field[i, :, :terminal_time]
                            self.gradientField.append(gradientField[i])
                            norm = np.linalg.norm(gradientField[i][:, 0])
                            self.robot_arrow = mpatches.FancyArrow(
                                self.robot_init_state[0],
                                self.robot_init_state[1],
                                gradientField[i][0, 0] * 0.1 * norm,
                                gradientField[i][1, 0] * 0.1 * norm,
                                width=0.05,
                                color=self.color_palette[i],
                            )
                            self.ax.add_patch(self.robot_arrow)

                        for i in range(num_obs):
                            self.docbf_arrow = mpatches.FancyArrow(
                                obs_grad_field[0][i][0][0],
                                obs_grad_field[0][i][0][1],
                                obs_grad_field[0][i][1][0] * 0.1,
                                obs_grad_field[0][i][1][1] * 0.1,
                                width=0.05,
                                color=self.color_palette[i],
                            )
                            self.ax.add_patch(self.docbf_arrow)

        def update_distance_field(frame, obstacle_elements, ax, line):

            if mode == "clf":
                pass

            if mode == "clf_cbf":
                pass

            if case_flag == 8:
                for element in obstacle_elements:
                    for coll in element.collections:
                        coll.remove()

                obstacle_elements.clear()  # Clear the list outside the loop
                cdf.obj_lists = []
                for i in range(num_obs):
                    cdf.obj_lists.append(
                        Circle(center=torch.from_numpy(obs_info[frame][i]), radius=0.3, device=device))
                d_grad, grad_plot = cdf.inference_c_space_sdf_using_data(cdf.Q_sets)
                contour, contourf, ct_zero, hatch_handle = cdf.plot_cdf_ax(d_grad.detach().cpu().numpy(), ax)
                obstacle_elements.extend([contour, contourf, ct_zero])

                if show_arrow:
                    for i in range(len(self.gradientField)):
                        norm = np.linalg.norm(self.gradientField[i][:, frame])
                        self.robot_arrow = mpatches.FancyArrow(
                            self.xt[0][frame],
                            self.xt[1][frame],
                            self.gradientField[i][0, frame] * 0.1 * norm,
                            self.gradientField[i][1, frame] * 0.1 * norm,
                            width=0.05,
                            color=self.color_palette[i],
                        )
                        self.ax.add_patch(self.robot_arrow)

                    for i in range(num_obs):
                        self.docbf_arrow = mpatches.FancyArrow(
                            obs_grad_field[frame][i][0][0],
                            obs_grad_field[frame][i][0][1],
                            obs_grad_field[frame][i][1][0] * 0.1,
                            obs_grad_field[frame][i][1][1] * 0.1,
                            width=0.05,
                            color=self.color_palette[i],
                        )
                        self.ax.add_patch(self.docbf_arrow)

            line.set_data(xt[0, :frame + 1], xt[1, :frame + 1])
            self.ax.set_xlim([-3.14, 3.14])
            self.ax.set_ylim([-3.14, 3.14])

            return obstacle_elements

        obstacle_elements = []
        # update_distance_field(0, obstacle_elements, self.ax, line)
        # self.ax.legend(handles=[hatch_handle], loc='upper center', ncol=2)
        # self.ax.legend([l1, l2], ['Start', 'goal'])
        # handles_all = [l1, l2, hatch_handle]
        # labels_all = [l.get_label() for l in handles_all]
        # self.ax.legend(handles=handles_all, labels=labels_all, loc='upper center', ncol=3)

        ani = FuncAnimation(self.fig, lambda frame: update_distance_field(frame, obstacle_elements, self.ax, line),
                            frames=num_frames, interval=50)
        if save_gif:
            writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            filename = os.path.join(save_path, distance_field + '_c_space' + str(num_obs) + 'obs.gif')
            ani.save(filename, writer=writer)

        plt.show()

    def render_ani_t_space_manipulator(self, distance_field, cdf, xt, terminal_time, reach_Mode=None, case_flag=None,
                                       obs_info=None, obs_list=None, save_gif=False, save_path=None):
        f_rob_start = \
            cdf.robot.forward_kinematics_all_joints(torch.from_numpy(xt[:2, 0]).to(device).unsqueeze(0))[
                0].detach().cpu().numpy()
        f_rob_end = \
            cdf.robot.forward_kinematics_all_joints(
                torch.from_numpy(self.robot_target_state[0:2]).to(device).unsqueeze(0))[
                0].detach().cpu().numpy()

        if case_flag == 5:
            # for i in range(num_obs):
            #     circle_plot = plt.Circle(cdf.obj_lists[0].center.detach().cpu().numpy(), 0.3, color='k', hatch='///',
            #                              fill=False,
            #                              label='Goal')
            #     self.ax.add_artist(circle_plot)
            log_point_pos = []
            log_point_grad = []
            log_point_dist = []
            log_point_robot = []
            sdf_obj, grad_obj, q_obj, q_robot = None, None, None, None
            for i in range(terminal_time):
                if distance_field == "sdf":
                    sdf_obj, grad_obj, q_obj, q_robot = cdf.inference_t_sdf_grad(
                        torch.from_numpy(xt[:2, i]).to(device).unsqueeze(0))
                    log_point_robot.append(q_robot.detach().cpu().numpy().flatten())
                elif distance_field == "cdf":
                    sdf_obj, grad_obj, q_obj = cdf.inference_t_space_sdf_using_data(
                        torch.from_numpy(xt[:2, i]).to(device).unsqueeze(0))
                sdf_obj.detach().cpu().numpy()
                grad_obj.detach().cpu().numpy()
                q_obj.detach().cpu().numpy()
                log_point_pos.append(q_obj.detach().cpu().numpy().flatten())
                log_point_grad.append(grad_obj.detach().cpu().numpy().flatten())
                log_point_dist.append(sdf_obj.detach().cpu().numpy())

        if case_flag == 7:
            log_point_pos = []
            log_point_grad = []
            log_point_dist = []
            log_point_robot = []
            sdf_obj, grad_obj, q_obj, q_robot = None, None, None, None
            for i in range(terminal_time):
                if distance_field == "sdf":
                    sdf_obj, grad_obj, q_obj, q_robot = cdf.inference_t_sdf_grad(
                        torch.from_numpy(xt[:2, i]).to(device).unsqueeze(0))
                    log_point_robot.append(q_robot.detach().cpu().numpy().flatten())
                elif distance_field == "cdf":
                    sdf_obj, grad_obj, q_obj = cdf.inference_t_space_sdf_using_data(
                        torch.from_numpy(xt[:2, i]).to(device).unsqueeze(0))
                sdf_obj.detach().cpu().numpy()
                grad_obj.detach().cpu().numpy()
                q_obj.detach().cpu().numpy()
                log_point_pos.append(q_obj.detach().cpu().numpy().flatten())
                log_point_grad.append(grad_obj.detach().cpu().numpy().flatten())
                log_point_dist.append(sdf_obj.detach().cpu().numpy())

        num_obs = len(obs_list)
        self.fig, self.ax = plt.subplots()

        def update(frame):
            self.ax.clear()
            # plot the log_point_pos
            if case_flag == 5:
                if distance_field == "sdf":
                    plt.plot(log_point_robot[frame][0], log_point_robot[frame][1], 'g*', markersize=8)
                    plt.plot(log_point_pos[frame][0], log_point_pos[frame][1], 'r*')
                    plt.quiver(log_point_pos[frame][0], log_point_pos[frame][1], log_point_grad[frame][0],
                               log_point_grad[frame][1], color='b', scale=10)
                elif distance_field == "cdf":
                    f_rob_traj = cdf.robot.forward_kinematics_all_joints(
                        torch.from_numpy(log_point_pos[frame]).to(device).unsqueeze(0))[0].detach().cpu().numpy()
                    plt.plot(f_rob_traj[0, :], f_rob_traj[1, :], linestyle='-', color='b', linewidth=2.0)

            if case_flag == 7:
                if reach_Mode == "point_to_point":
                    plt.plot(f_rob_end[0, :], f_rob_end[1, :], linestyle='-', color='b', linewidth=2.0, label='Goal')
                    for i in range(num_obs):
                        circle_plot = plt.Circle(cdf.obj_lists[0].center.detach().cpu().numpy(), obs_list[i].radius,
                                                 color='k', hatch='///', fill=False, label='Goal')
                        self.ax.add_artist(circle_plot)

            if case_flag == 8:
                if reach_Mode == "point_to_point":
                    plt.plot(f_rob_end[0, :], f_rob_end[1, :], linestyle='-', color='b', linewidth=2.0, label='Goal')
                for i in range(num_obs):
                    circle_plot = plt.Circle(obs_info[frame][i], obs_list[i].radius, color='k', hatch='///', fill=False)
                    self.ax.add_artist(circle_plot)

            "plot the start and end points"
            plt.plot(f_rob_start[0, :], f_rob_start[1, :], linestyle='-', color='r', linewidth=2.0, label='Start')
            self.plot_2d_manipulators(joint_angles_batch=xt[:2, frame].reshape(1, 2))
            plt.legend(loc='upper center', ncol=3)
            self.ax.set_xlim([-4.5, 4.5])
            self.ax.set_ylim([-4.5, 4.5])
            self.ax.set_aspect('equal')

        num_frames = terminal_time
        ani = FuncAnimation(self.fig, update, frames=num_frames, interval=50)

        if save_gif:
            writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            # filename = os.join(save_path, distance_field + '_' + str(num_obs) + '.gif')
            filename = os.path.join(save_path, distance_field + '_t_space' + str(num_obs) + 'obs.gif')
            ani.save(filename, writer=writer)
        plt.show()

    def render_ani_manipulator(self, cdf, obs_center, xt, num_obs, terminal_time):
        f_rob_start = \
            cdf.robot.forward_kinematics_all_joints(torch.from_numpy(xt[:2, 0]).to(device).unsqueeze(0))[
                0].detach().cpu().numpy()
        f_rob_end = \
            cdf.robot.forward_kinematics_all_joints(
                torch.from_numpy(self.robot_target_state[0:2]).to(device).unsqueeze(0))[
                0].detach().cpu().numpy()

        self.fig, self.ax = plt.subplots()

        def update(frame):
            self.ax.clear()
            # plot the start and end points
            for i in range(num_obs):
                circle_plot = plt.Circle(obs_center[frame][0], 0.3, color='k', hatch='///', fill=False,
                                         label='Obstacle')
                self.ax.add_artist(circle_plot)

            plt.plot(f_rob_start[0, :], f_rob_start[1, :], linestyle='-', color='r', linewidth=2.0, label='Start')
            plt.plot(f_rob_end[0, :], f_rob_end[1, :], linestyle='-', color='b', linewidth=2.0, label='Goal')
            self.plot_2d_manipulators(joint_angles_batch=xt[:2, frame].reshape(1, 2))
            plt.legend(loc='upper center', ncol=3)
            self.ax.set_xlim([-4.5, 4.5])
            self.ax.set_ylim([-4.5, 4.5])
            self.ax.set_aspect('equal')

        num_frames = terminal_time
        ani = FuncAnimation(self.fig, update, frames=num_frames, interval=50)
        plt.show()

    def render_c_space(self, distance_field, cdf, xt, terminal_time, case_flag=None):

        if case_flag == 7:
            if distance_field == 'sdf':
                print("Start to visualize the c space of SDF")
                cdf.plot_sdf()
            elif distance_field == 'cdf':
                print("Start to visualize the c space of CDF")
                d_plot, grad_plot = cdf.inference_c_space_sdf_using_data(cdf.Q_sets)
                cdf.plot_cdf(d_plot.detach().cpu().numpy(), grad_plot.detach().cpu().numpy())

            "plot the start and goal point of the robot in configuration space"
            plt.scatter(self.robot_init_state[0], self.robot_init_state[1], color='g', s=25, zorder=10, label='Start')
            plt.scatter(self.robot_target_state[0], self.robot_target_state[1], color='r', s=25, zorder=10,
                        label='Goal')

            "plot the trajectory of the robot in configuration space"
            xt = xt[:, :terminal_time]
            plt.plot(xt[0, :], xt[1, :], color='yellow', linestyle='-', linewidth=2.0, label='Trajectory')
            plt.legend(loc='upper center', ncol=3)

        if case_flag == 5:
            if distance_field == 'sdf':
                print("Start to visualize the c space of SDF")
                cdf.plot_sdf()
            elif distance_field == 'cdf':
                print("Start to visualize the c space of CDF")
                d_plot, grad_plot = cdf.inference_c_space_sdf_using_data(cdf.Q_sets)
                cdf.plot_cdf(d_plot.detach().cpu().numpy(), grad_plot.detach().cpu().numpy())

            "plot the start and goal point of the robot in configuration space"
            plt.scatter(self.robot_init_state[0], self.robot_init_state[1], color='g', s=25, zorder=10, label='Start')
            target_state_config = \
                cdf.robot.forward_kinematics_all_joints(
                    torch.from_numpy(self.robot_target_state[0:2]).to(device).unsqueeze(0))[
                    0].detach().cpu().numpy()

            "plot the trajectory of the robot in configuration space"
            xt = xt[:, :terminal_time]
            plt.plot(xt[0, :], xt[1, :], color='y', linestyle='-', linewidth=2.0, label='Trajectory')
            plt.legend(loc='upper center', ncol=3)

        plt.show()

    def render_manipulator(self, cdf, xt, terminal_time, case_flag=None):
        self.fig, self.ax = plt.subplots()
        xf_2d = self.robot_target_state[0:2]
        plt.rcParams['axes.facecolor'] = '#eaeaf2'
        ax = plt.gca()

        f_rob_start = \
            cdf.robot.forward_kinematics_all_joints(torch.from_numpy(xt[:2, 0]).to(device).unsqueeze(0))[
                0].detach().cpu().numpy()
        f_rob_end = \
            cdf.robot.forward_kinematics_all_joints(torch.from_numpy(xt[:2, terminal_time]).to(device).unsqueeze(0))[
                0].detach().cpu().numpy()

        "Plot the base of the 2D manipulator"
        circle = plt.Circle((0, 0), 2, color='r', fill=True, linestyle='--', linewidth=1.0, zorder=10, alpha=0.1)
        ax.add_artist(circle)
        plt.plot(0, 0, marker='o', markersize=15, zorder=10, markerfacecolor='#DDA15E', markeredgecolor='k')
        "Plot the initial state"
        plt.plot(f_rob_start[0, :], f_rob_start[1, :], linestyle='-', color='r', zorder=5, linewidth=2.0, label='Start')
        "Plot the final state"
        plt.plot(f_rob_end[0, :], f_rob_end[1, :], linestyle='-', color='b', zorder=5, linewidth=2.0, label='End')
        "Plot the manipulator trajectory"
        manipulator_angles = (xt[0:2, :terminal_time]).T
        self.plot_2d_manipulators(joint_angles_batch=manipulator_angles)
        if case_flag == 1:
            pass
        elif case_flag == 2:
            pass
        elif case_flag == 3:
            print("The current case is 3")
            "Plot the target(goal) position in task space"
            plt.scatter(self.robot_target_state[0], self.robot_target_state[1], color='k', s=50, zorder=5,
                        label='Goal')
            plt.legend(loc='lower left', ncol=3)
        elif case_flag == 5:
            print("The current case is 5")
            "It's a whole body manipulation task, we need to visualize the object"
            cdf.obj_lists[0].label = 'Goal'
            self.ax.add_patch(cdf.obj_lists[0].create_patch())
            plt.legend(loc='lower left', ncol=3)
        elif case_flag == 6:
            pass
        elif case_flag == 7:
            print("The current case is 7")
            cdf.obj_lists[0].label = 'Obstacle'
            self.ax.add_patch(cdf.obj_lists[0].create_patch())
            plt.legend(loc='lower left', ncol=3)

        ax.set_aspect('equal')
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.show()

    @staticmethod
    def plot_2d_manipulators(link1_length=2.0, link2_length=2.0, joint_angles_batch=None):
        # Check if joint_angles_batch is None or has incorrect shape
        if joint_angles_batch is None or joint_angles_batch.shape[1] != 2:
            raise ValueError("joint_angles_batch must be provided with shape (N, 2)")

        # Number of sets of joint angles
        num_sets = joint_angles_batch.shape[0]

        # Create a figure
        cmap = cm.get_cmap('Greens', num_sets)  # You can choose other colormaps like 'Greens', 'Reds', etc.
        cmap2 = cm.get_cmap('Reds', num_sets)  # You can choose other colormaps like 'Greens', 'Reds', etc.
        # the color will
        for i in range(num_sets):
            # Extract joint angles for the current set
            theta1, theta2 = joint_angles_batch[i]

            # Calculate the position of the first joint
            joint1_x = link1_length * np.cos(theta1)
            joint1_y = link1_length * np.sin(theta1)

            # Calculate the position of the end effector (tip of the second link)
            end_effector_x = joint1_x + link2_length * np.cos(theta1 + theta2)
            end_effector_y = joint1_y + link2_length * np.sin(theta1 + theta2)

            # Stack the base, joint, and end effector positions
            positions = np.vstack([[0, 0], [joint1_x, joint1_y], [end_effector_x, end_effector_y]])  # shape: (3, 2)

            # Plotting
            plt.plot(positions[:, 0], positions[:, 1], linestyle='-', color='green', marker='o', markersize=5,
                     markerfacecolor='white',
                     markeredgecolor='green', alpha=0.3)

            # cover the end effector with different colors to hightlight the trajectory
            plt.plot(positions[2, 0], positions[2, 1], linestyle='-', color=cmap(i), marker='o', markersize=5,
                     markerfacecolor='white',
                     markeredgecolor=cmap2(i))
            # plot a bigger base center at (0, 0), which is a cirlce with golden color
            plt.plot(0, 0, marker='o', markersize=15, markerfacecolor='#DDA15E', markeredgecolor='k')

    def show_clf(self, clft, terminal_time, save_result=False, save_path=None):
        """ show changes in clf """
        t = np.arange(0, terminal_time * self.dt, self.dt)[0:terminal_time]

        # add dpi
        figure = plt.figure()

        plt.plot(t, clft[0:terminal_time].reshape(terminal_time, ), linewidth=3, color='red')
        plt.title('Changes in CLF', self.label_font)
        plt.ylabel('V(x)', self.label_font)
        plt.xlabel('Time (s)', self.label_font)

        # set the tick in Times New Roman and size
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        plt.grid()
        if save_result:
            file_name = os.path.join(save_path, 'clf.png')
            plt.savefig(file_name, format='png')
        # plt.savefig('clf.png', format='png', dpi=300)
        plt.show()

    def show_slack(self, slackt, terminal_time, save_result=False, save_path=None):
        """ show changes in clf """
        t = np.arange(0, terminal_time * self.dt, self.dt)[0:terminal_time]

        # add dpi
        figure = plt.figure()

        plt.plot(t, slackt[0:terminal_time].reshape(terminal_time, ), linewidth=3, color='blue')
        plt.title('Changes in slack variable', self.label_font)
        plt.ylabel('Slack Variable', self.label_font)
        plt.xlabel('Time (s)', self.label_font)

        # set the tick in Times New Roman and size
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        plt.grid()
        if save_result:
            file_name = os.path.join(save_path, 'slack.png')
            plt.savefig(file_name, format='png')
        # plt.savefig('slack.png', format='png', dpi=300)
        plt.show()

    def show_cbf(self, cbft, terminal_time, save_result=False, save_path=None):
        """ show changes in cbf """
        t = np.arange(0, terminal_time * self.dt, self.dt)[0:terminal_time]

        num_obs = cbft.shape[0]
        # add dpi
        figure = plt.figure()
        self.ax = plt.gca()
        # figure.set_dpi(300)

        for i in range(num_obs):
            plt.plot(t, cbft[i, 0, 0:terminal_time].reshape(terminal_time, ), linewidth=3,
                     label='obstacle {}'.format(i))

        plt.plot(t, np.zeros(t.shape[0]), color='k', linestyle='-', linewidth=2.0, label='zero level')

        plt.title('CBF with respect to {} obstacles'.format(num_obs), self.label_font)
        plt.ylabel('cbf (m)', self.label_font)
        plt.xlabel('Time (s)', self.label_font)
        # set the tick in Times New Roman and size
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        plt.grid()
        plt.legend(ncol=num_obs + 1, prop=self.legend_font)
        if save_result:
            file_name = os.path.join(save_path, 'cbf.png')
            plt.savefig(file_name, format='png', bbox_inches='tight')
        plt.show()

    def show_dx_cbf(self, i, dxcbft, terminal_time, save_result=False, save_path=None):
        """ show changes in dxcbf """
        t = np.arange(0, terminal_time * self.dt, self.dt)[0:terminal_time]

        # add dpi
        figure = plt.figure()

        plt.plot(t, dxcbft[i, 0, 0:terminal_time].reshape(terminal_time, ), linewidth=3, color='blue', label='x')
        plt.plot(t, dxcbft[i, 1, 0:terminal_time].reshape(terminal_time, ), linewidth=3, color='red', label='y')
        plt.title('dxcbf with respect to {}th obstacle'.format(i), self.label_font)
        plt.ylabel('dxcbf (m/s)', self.label_font)
        plt.xlabel('Time (s)', self.label_font)

        # set the tick in Times New Roman and size
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        plt.legend(loc='upper right', prop=self.legend_font)
        plt.grid()
        if save_result:
            file_name = os.path.join(save_path, 'dxcbf.png')
            plt.savefig(file_name, format='png', dpi=300)
        plt.show()

    def show_integral_controls(self, ut, terminal_time, save_result=False, save_path=None):
        """ show controls of integral model """
        t = np.arange(0, terminal_time * self.dt, self.dt)[0:terminal_time]

        # add dpi
        figure = plt.figure()

        plt.plot(t, ut[0][0:terminal_time].reshape(terminal_time, ), linewidth=3, color="b", label="vx", )
        plt.plot(t, self.umax[0] * np.ones(t.shape[0]), color="b", linestyle='dashed')
        plt.plot(t, self.umin[0] * np.ones(t.shape[0]), color="b", linestyle='dashed')

        plt.plot(t, ut[1][0:terminal_time].reshape(terminal_time, ), linewidth=3, color="r", label="vy", )
        plt.title("Control Variables", self.label_font)
        plt.xlabel("Time (s)", self.label_font)
        plt.ylabel("vx (m/s) / vy (m/s)", self.label_font)

        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname("Times New Roman") for label in labels]

        plt.legend(loc="upper right", prop=self.legend_font)
        plt.grid()
        if save_result:
            file_name = os.path.join(save_path, "controls.png")
            plt.savefig(file_name, format="png", dpi=300)
        # plt.savefig("controls.png", format="png", dpi=300)
        plt.show()
