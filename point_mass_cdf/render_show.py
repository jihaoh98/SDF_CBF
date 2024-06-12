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
    def __init__(self, robot_params, cir_obs_params, dt) -> None:
        """ init the render animation """
        self.dt = dt

        # robot
        self.robot_init_state = np.array(robot_params['initial_state'])
        self.robot_target_state = np.array(robot_params['target_state'])
        self.robot_radius = robot_params['radius']
        self.umax = robot_params['u_max']
        self.umin = robot_params['u_min']

        # obstacle
        if cir_obs_params is not None:
            self.cir_obs_num = len(cir_obs_params['obs_states'])
            self.cir_obs = [None for i in range(self.cir_obs_num)]

        # storage the past states of robots and different shaped obs
        self.xt = None
        self.cir_obs_list_t = None

        # plot
        self.fig, self.ax = plt.subplots()

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
        self.legend_font = {"family": "Times New Roman", "weight": "normal", "size": 12}

    def render_cdf(self, i, cdf, xt, obs_list, terminal_time, show_obs, dxcbft, save_gif=False, show_arrow=False):
        # visualize the cdf field
        color_palette = ['k', 'purple', 'y']
        self.color_palette = color_palette
        cdf.obj_lists = obs_list
        cdf.obj_lists = []
        cdf.q_template = torch.load(os.path.join(CUR_PATH, 'data2D_100.pt'))

        # plot the colorful zero level set
        for i in range(len(obs_list)):
            cdf.obj_lists = [
                Circle(center=torch.from_numpy(obs_list[i].state), radius=obs_list[i].radius, device=device)]
            d, grad = cdf.inference_c_space_sdf_using_data(cdf.Q_sets, 60)
            cdf.plot_cdf(d.detach().cpu().numpy(), grad.detach().cpu().numpy(), color=color_palette[i])
        # plot the unit other level set
        cdf.obj_lists = []
        for i in range(len(obs_list)):
            cdf.obj_lists.append(
                Circle(center=torch.from_numpy(obs_list[i].state), radius=obs_list[i].radius, device=device))
            d, grad = cdf.inference_c_space_sdf_using_data(cdf.Q_sets, 60)
            cdf.plot_non_zero_cdf(d.detach().cpu().numpy(), grad.detach().cpu().numpy())

        """ Visualization """
        self.fig.set_size_inches(7, 6.5)
        self.fig.set_dpi(150)
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-3.14, 3.14)
        self.ax.set_ylim(-3.14, 3.14)
        self.ax.set_xlabel('x (m)', self.label_font)
        self.ax.set_ylabel("y (m)", self.label_font)
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        self.xt = xt
        self.show_obs = show_obs
        self.show_arrow = show_arrow
        self.animation_init()
        # robot and the arrow
        self.robot_body = mpatches.Circle(
            (self.robot_init_state[0], self.robot_init_state[1]),
            radius=self.robot_radius,
            edgecolor='silver',
            fill=False
        )
        self.ax.add_patch(self.robot_body)

        if show_arrow:
            gradientField = np.zeros((len(obs_list), 3, terminal_time))
            for i in range(len(obs_list)):
                gradientField[i] = dxcbft[i, :, :]
                self.gradientField.append(gradientField[i])
                norm = np.linalg.norm(gradientField[i][:, 0])
                self.robot_arrow = mpatches.FancyArrow(
                    self.robot_init_state[0],
                    self.robot_init_state[1],
                    gradientField[i][0, 0] * 0.05 * norm,
                    gradientField[i][1, 0] * 0.05 * norm,
                    width=0.025,
                    color=color_palette[i],
                )
                self.ax.add_patch(self.robot_arrow)

        self.ani = animation.FuncAnimation(
            self.fig,
            func=self.animation_loop_cdf,
            frames=terminal_time,
            init_func=self.animation_init,
            interval=200,
            repeat=False,
        )
        if save_gif:
            writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            file_path = os.path.join(CUR_PATH, 'integral.gif')
            self.ani.save(file_path, writer=writer)
        plt.show()

    def render_manipulator(self, cdf, xt, terminal_time):
        plt.rcParams['axes.facecolor'] = '#eaeaf2'
        ax = plt.gca()

        for obj in cdf.obj_lists:
            ax.add_patch(obj.create_patch())

        xf_2d = self.robot_target_state[0:2]
        manipulator_angles = (xt[0:2, :terminal_time]).T
        self.plot_2d_manipulators(joint_angles_batch=manipulator_angles)
        f_rob_end = cdf.robot.forward_kinematics_all_joints(torch.from_numpy(xf_2d).to(device).unsqueeze(0))[
            0].detach().cpu().numpy()
        plt.scatter(f_rob_end[0, -1], f_rob_end[1, -1], color='r', s=100, zorder=10, label='Goal')
        ax.set_aspect('equal')
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.legend(loc='upper left')
        plt.show()

    def plot_2d_manipulators(self, link1_length=2.0, link2_length=2.0, joint_angles_batch=None):
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

    def render_dynamic_cdf(self, cdf, log_circle_center, log_gradient_field, xt, terminal_time, show_obs, dxcbft,
                           obs_num, save_gif=False, show_arrow=False, show_ob_arrow=False):
        cdf.q_template = torch.load(os.path.join(CUR_PATH, 'data2D_100.pt'))
        cdf.obj_lists = None
        line, = self.ax.plot([], [], color='yellow', linestyle='--', linewidth=2)

        # plot the start and goal point of the robot
        self.ax.plot(xt[0, 0], xt[1, 0], 'r*', label='Start')
        self.ax.plot(self.robot_target_state[0], self.robot_target_state[1], 'r*', label='Goal')

        num_obs = obs_num
        self.show_arrow = show_arrow
        self.show_ob_arrow = show_ob_arrow
        self.xt = xt

        if self.show_arrow:
            gradientField = dxcbft[0, :, :]  # shape is (2, time_steps)
            self.gradientField = gradientField
            norm = np.linalg.norm(gradientField[:, 0])
            self.robot_arrow = mpatches.FancyArrow(
                self.robot_init_state[0],
                self.robot_init_state[1],
                self.gradientField[0, 0] * 0.1 * norm,
                self.gradientField[1, 0] * 0.1 * norm,
                width=0.05,
                color='k',
            )
            self.ax.add_patch(self.robot_arrow)

        if self.show_ob_arrow:
            self.docbf_arrow = mpatches.FancyArrow(
                log_gradient_field[0][0],
                log_gradient_field[0][1],
                log_gradient_field[0][2] * 0.1,
                log_gradient_field[0][3] * 0.1,
                width=0.05,
                color='r',
            )
            self.ax.add_patch(self.docbf_arrow)

        def update_distance_field(frame, obstacle_elements, ax, line):

            # re-update the obstacle
            if num_obs == 1:
                for element in obstacle_elements:
                    for coll in element.collections:
                        coll.remove()

                obstacle_elements.clear()  # Clear the list outside the loop
                object_center = log_circle_center[frame]
                cdf.obj_lists = [Circle(center=torch.from_numpy(object_center), radius=0.3, device=device)]
                d_grad, grad_plot = cdf.inference_c_space_sdf_using_data(cdf.Q_sets)
                # plot the distance field
                contour, contourf, ct_zero = cdf.plot_cdf_ax(d_grad.detach().cpu().numpy(), ax)
                # Add new elements to the list
                obstacle_elements.extend([contour, contourf, ct_zero])

            elif num_obs == 2:
                for element in obstacle_elements:
                    for coll in element.collections:
                        coll.remove()

                obstacle_elements.clear()
                cdf.obj_lists = [
                    Circle(center=torch.from_numpy(log_circle_center[frame][0]), radius=0.3, device=device),
                    Circle(center=torch.from_numpy(log_circle_center[frame][1]), radius=0.3, device=device)]
                d_grad, grad_plot = cdf.inference_c_space_sdf_using_data(cdf.Q_sets, self.sample_num)
                contour, contourf, ct_zero = cdf.plot_cdf_ax(d_grad.detach().cpu().numpy(), ax)
                # Add new elements to the list
                obstacle_elements.extend([contour, contourf, ct_zero])

            if self.show_arrow:
                norm = np.linalg.norm(self.gradientField[:, frame])
                self.robot_arrow = mpatches.FancyArrow(
                    self.xt[:, frame][0],
                    self.xt[:, frame][1],
                    self.gradientField[0, frame] * 0.1 * norm,
                    self.gradientField[1, frame] * 0.1 * norm,
                    width=0.05,
                    color='k',
                )
                self.ax.add_patch(self.robot_arrow)

            if self.show_ob_arrow:
                self.docbf_arrow = mpatches.FancyArrow(
                    log_gradient_field[frame][0],
                    log_gradient_field[frame][1],
                    log_gradient_field[frame][2] * 0.1,
                    log_gradient_field[frame][3] * 0.1,
                    width=0.05,
                    color='r',
                )
                self.ax.add_patch(self.docbf_arrow)

            line.set_data(xt[0, :frame + 1], xt[1, :frame + 1])

            return obstacle_elements

        obstacle_elements = []
        # plt.legend(loc='upper center', ncols=2)
        num_frames = terminal_time
        ani = FuncAnimation(self.fig, lambda frame: update_distance_field(frame, obstacle_elements, self.ax, line),
                            frames=num_frames, interval=50)
        plt.show()

    def render(self, i, xt, cir_obs_list_t, terminal_time, show_obs, dxcbft, docbft, save_gif=False):
        # dxcbft: shape is (cir_obs_num, 2, time_steps)
        gradientField = dxcbft[i, :, :]  # shape is (2, time_steps)
        docbfield = docbft[i, :, :]  # shape is (2, time_steps)

        self.gradientField = gradientField
        self.docbfield = docbfield

        """ Visualization """
        self.fig.set_size_inches(7, 6.5)
        self.fig.set_dpi(150)
        self.ax.set_aspect('equal')

        self.ax.set_xlim(-5, 5.0)
        self.ax.set_ylim(-5, 5.0)

        self.ax.set_xlabel('x (m)', self.label_font)
        self.ax.set_ylabel("y (m)", self.label_font)

        # set the tick in Times New Roman and size
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        self.xt = xt
        self.cir_obs_list_t = cir_obs_list_t
        self.show_obs = show_obs

        self.animation_init()

        # robot and the arrow
        self.robot_body = mpatches.Circle(
            (self.robot_init_state[0], self.robot_init_state[1]),
            radius=self.robot_radius,
            edgecolor='silver',
            fill=False
        )
        self.ax.add_patch(self.robot_body)

        # use the gradeint field to show the direction of the robot
        # self.robot_arrow = mpatches.Arrow(
        #     self.robot_init_state[0],
        #     self.robot_init_state[1],
        #     self.gradientField[0, i],
        #     self.gradientField[1, i],
        #     width=0.05,
        #     color='k',
        # )
        # self.ax.add_patch(self.robot_arrow)

        norm = np.linalg.norm(gradientField[:, 0])
        norm2 = np.linalg.norm(docbfield[:, 0])

        self.robot_arrow = mpatches.FancyArrow(
            self.robot_init_state[0],
            self.robot_init_state[1],
            self.gradientField[0, 0] * 0.05 * norm,
            self.gradientField[1, 0] * 0.05 * norm,
            width=0.05,
            color='k',
        )
        self.ax.add_patch(self.robot_arrow)

        self.docbf_arrow = mpatches.FancyArrow(
            cir_obs_list_t[i][0, 0],
            cir_obs_list_t[i][1, 0],
            self.docbfield[0, 0] * 0.05 * norm2,
            self.docbfield[1, 0] * 0.05 * norm2,
            width=0.025,
            color='r',
        )
        self.ax.add_patch(self.docbf_arrow)

        # self.robot_arrow = mpatches.Arrow(
        #     self.robot_init_state[0],
        #     self.robot_init_state[1],
        #     self.robot_width * np.cos(self.robot_init_state[2]),
        #     self.robot_width * np.sin(self.robot_init_state[2]),
        #     width=0.15,
        #     color='k',
        # )
        # self.ax.add_patch(self.robot_arrow)

        # show obstacles
        if self.show_obs:
            if self.cir_obs_list_t is not None:
                for i in range(self.cir_obs_num):
                    self.cir_obs[i] = mpatches.Circle(
                        xy=self.cir_obs_list_t[i][0:2, 0],
                        radius=self.cir_obs_list_t[i][4, 0],
                        edgecolor='k',
                        fill=False
                    )
                    self.ax.add_patch(self.cir_obs[i])

        self.ani = animation.FuncAnimation(
            self.fig,
            func=self.animation_loop,
            frames=terminal_time + 1,
            init_func=self.animation_init,
            interval=200,
            repeat=False,
        )

        plt.grid('--')
        if save_gif:
            writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            self.ani.save('integral.gif', writer=writer)
        plt.show()

    def animation_init(self):
        """ init the robot start and end position """
        # start body and target body
        self.start_body, = plt.plot(self.robot_init_state[0], self.robot_init_state[1], color='y', marker='*',
                                    markersize=15)
        self.end_body, = plt.plot(self.robot_target_state[0], self.robot_target_state[1], color='r', marker='*',
                                  markersize=15)

        return self.ax.patches + self.ax.texts + self.ax.artists

    def animation_loop_cdf(self, indx):
        """ loop for update the position of robot and obstacles """
        # robot
        self.robot_body.remove()
        self.robot_body = mpatches.Circle(xy=self.xt[:, indx][0:2], radius=self.robot_radius, edgecolor='r', fill=False)
        self.ax.add_patch(self.robot_body)

        if self.show_arrow:
            for i in range(len(self.gradientField)):
                norm = np.linalg.norm(self.gradientField[i][:, indx])
                self.robot_arrow = mpatches.FancyArrow(
                    self.xt[0][indx],
                    self.xt[1][indx],
                    self.gradientField[i][0, indx] * 0.05 * norm,
                    self.gradientField[i][1, indx] * 0.05 * norm,
                    width=0.025,
                    color=self.color_palette[i],
                )
                self.ax.add_patch(self.robot_arrow)

        # show past trajecotry of robot and obstacles
        if indx != 0:
            x_list = [self.xt[:, indx - 1][0], self.xt[:, indx][0]]
            y_list = [self.xt[:, indx - 1][1], self.xt[:, indx][1]]
            self.ax.plot(x_list, y_list, color='b', )

            # # show past trajecotry of each dynamic obstacle
            # if self.show_obs:
            #     if self.cir_obs_list_t is not None:
            #         for i in range(self.cir_obs_num):
            #             ox_list = [self.cir_obs_list_t[i][:, indx - 1][0], self.cir_obs_list_t[i][:, indx][0]]
            #             oy_list = [self.cir_obs_list_t[i][:, indx - 1][1], self.cir_obs_list_t[i][:, indx][1]]
            #             self.ax.plot(ox_list, oy_list, linestyle='--', color='k', )

        # plt.savefig('figure/{}.png'.format(indx), format='png', dpi=300)
        return self.ax.patches + self.ax.texts + self.ax.artists

    def animation_loop(self, indx):
        """ loop for update the position of robot and obstacles """
        # robot
        self.robot_body.remove()
        self.robot_body = mpatches.Circle(xy=self.xt[:, indx][0:2], radius=self.robot_radius, edgecolor='r', fill=False)
        self.ax.add_patch(self.robot_body)

        norm = np.linalg.norm(self.gradientField[:, indx])
        self.robot_arrow = mpatches.FancyArrow(
            self.xt[:, indx][0],
            self.xt[:, indx][1],
            self.gradientField[0, indx] * 0.1 * norm,
            self.gradientField[1, indx] * 0.1 * norm,
            width=0.05,
            color='k',
        )
        self.ax.add_patch(self.robot_arrow)
        norm2 = np.linalg.norm(self.docbfield[:, indx])
        self.docbf_arrow = mpatches.FancyArrow(
            self.cir_obs_list_t[0][0, indx],
            self.cir_obs_list_t[0][1, indx],
            self.docbfield[0, indx] * 0.1 * norm2,
            self.docbfield[1, indx] * 0.1 * norm2,
            width=0.025,
            color='r',
        )
        self.ax.add_patch(self.docbf_arrow)

        # obs
        if self.show_obs:
            if self.cir_obs_list_t is not None:
                for i in range(self.cir_obs_num):
                    self.cir_obs[i].remove()
                    self.cir_obs[i] = mpatches.Circle(
                        xy=self.cir_obs_list_t[i][:, indx][0:2],
                        radius=self.cir_obs_list_t[i][:, indx][4],
                        edgecolor='k',
                        fill=False
                    )
                    self.ax.add_patch(self.cir_obs[i])

                    # plot the center of the circle obstacle
                    self.ax.plot(self.cir_obs_list_t[i][:, indx][0], self.cir_obs_list_t[i][:, indx][1], 'k*')

                    # show past trajecotry of robot and obstacles
        if indx != 0:
            x_list = [self.xt[:, indx - 1][0], self.xt[:, indx][0]]
            y_list = [self.xt[:, indx - 1][1], self.xt[:, indx][1]]
            self.ax.plot(x_list, y_list, color='b', )

            # show past trajecotry of each dynamic obstacle
            if self.show_obs:
                if self.cir_obs_list_t is not None:
                    for i in range(self.cir_obs_num):
                        ox_list = [self.cir_obs_list_t[i][:, indx - 1][0], self.cir_obs_list_t[i][:, indx][0]]
                        oy_list = [self.cir_obs_list_t[i][:, indx - 1][1], self.cir_obs_list_t[i][:, indx][1]]
                        self.ax.plot(ox_list, oy_list, linestyle='-.', color='k', )

        # plt.savefig('figure/{}.png'.format(indx), format='png', dpi=300)
        return self.ax.patches + self.ax.texts + self.ax.artists

    def show_clf(self, clft, terminal_time):
        """ show changes in clf """
        t = np.arange(0, terminal_time * self.dt, self.dt)[0:terminal_time]

        # add dpi
        figure = plt.figure()
        figure.set_dpi(150)

        plt.plot(t, clft[0:terminal_time].reshape(terminal_time, ), linewidth=3, color='red')
        plt.title('Changes in CLF', self.label_font)
        plt.ylabel('V(x)', self.label_font)
        plt.xlabel('Time (s)', self.label_font)

        # set the tick in Times New Roman and size
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        plt.grid()
        # plt.savefig('clf.png', format='png', dpi=300)
        plt.show()

    def show_slack(self, slackt, terminal_time):
        """ show changes in clf """
        t = np.arange(0, terminal_time * self.dt, self.dt)[0:terminal_time]

        # add dpi
        figure = plt.figure()
        figure.set_dpi(150)

        plt.plot(t, slackt[0:terminal_time].reshape(terminal_time, ), linewidth=3, color='blue')
        plt.title('Changes in slack variable', self.label_font)
        plt.ylabel('Slack Variable', self.label_font)
        plt.xlabel('Time (s)', self.label_font)

        # set the tick in Times New Roman and size
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        plt.grid()
        # plt.savefig('slack.png', format='png', dpi=300)
        plt.show()

    def show_cdf_cbf(self, i, cdfcbft, terminal_time):
        """ show changes in cdfcbf """
        t = np.arange(0, terminal_time * self.dt, self.dt)[0:terminal_time]

        # add dpi
        figure = plt.figure()
        figure.set_dpi(150)

        plt.plot(t, cdfcbft[i, 0:terminal_time].reshape(terminal_time, ), linewidth=3, color='blue')
        plt.title('CDF-CBF with respect to {}th obstacle'.format(i), self.label_font)
        plt.ylabel('cdfcbf (m)', self.label_font)
        plt.xlabel('Time (s)', self.label_font)

        # set the tick in Times New Roman and size
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        plt.grid()
        # plt.savefig('cdfcbf.png', format='png', dpi=300)
        plt.show()

    def show_cbf(self, i, cbft, terminal_time):
        """ show changes in cbf """
        t = np.arange(0, terminal_time * self.dt, self.dt)[0:terminal_time]

        # add dpi
        figure = plt.figure()
        figure.set_dpi(150)

        plt.plot(t, cbft[i, 0:terminal_time].reshape(terminal_time, ), linewidth=3, color='blue')
        plt.title('CBF with respect to {}th obstacle'.format(i), self.label_font)
        plt.ylabel('cbf (m)', self.label_font)
        plt.xlabel('Time (s)', self.label_font)

        # set the tick in Times New Roman and size
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        plt.grid()
        # plt.savefig('cbf.png', format='png', dpi=300)
        plt.show()

    def show_dx_cbf(self, i, dxcbft, terminal_time):
        """ show changes in dxcbf """
        t = np.arange(0, terminal_time * self.dt, self.dt)[0:terminal_time]

        # add dpi
        figure = plt.figure()
        figure.set_dpi(150)

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
        # plt.savefig('dxcbf.png', format='png', dpi=300)
        plt.show()

    def show_integral_controls(self, ut, terminal_time):
        """ show controls of integral model """
        t = np.arange(0, terminal_time * self.dt, self.dt)[0:terminal_time]

        # add dpi
        figure = plt.figure()
        figure.set_dpi(150)

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
        # plt.savefig("controls.png", format="png", dpi=300)
        plt.show()
