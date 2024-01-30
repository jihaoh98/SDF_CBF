import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import numpy as np


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

        self.show_obs = True

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

    def render(self, xt, cir_obs_list_t, terminal_time, show_obs, cdf=None, save_gif=False):
        """ Visualization """
        self.fig.set_size_inches(7, 6.5)
        self.fig.set_dpi(150)
        self.ax.set_aspect('equal')

        self.ax.set_xlim(-4, 4.0)
        self.ax.set_ylim(-4, 4.0)

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

        # show cdf obstacles
        d_grad, grad_plot = cdf.inference_c_space_sdf_using_data(cdf.Q_sets)
        cdf.plot_cdf(d_grad.detach().cpu().numpy())

        self.ani = animation.FuncAnimation(
            self.fig,
            func=self.animation_loop,
            frames=terminal_time + 1,
            init_func=self.animation_init,
            interval=20,
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
        self.start_body, = plt.plot(self.robot_init_state[0], self.robot_init_state[0], color='purple', marker='*',
                                    markersize=8)
        self.end_body, = plt.plot(self.robot_target_state[0], self.robot_target_state[1], color='purple', marker='*',
                                  markersize=8)

        return self.ax.patches + self.ax.texts + self.ax.artists

    def animation_loop(self, indx):
        """ loop for update the position of robot and obstacles """
        # robot
        self.robot_body.remove()
        self.robot_body = mpatches.Circle(xy=self.xt[:, indx][0:2], radius=self.robot_radius, edgecolor='r', fill=False)
        self.ax.add_patch(self.robot_body)

        # self.robot_arrow.remove()
        # self.robot_arrow = mpatches.Arrow(
        #     self.xt[:, indx][0],
        #     self.xt[:, indx][1],
        #     self.robot_width * np.cos(self.xt[:, indx][2]),
        #     self.robot_width * np.sin(self.xt[:, indx][2]),
        #     width=0.15, 
        #     color='k',
        # )
        # self.ax.add_patch(self.robot_arrow)

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

                    # show past trajecotry of robot and obstacles
        if indx != 0:
            x_list = [self.xt[:, indx - 1][0], self.xt[:, indx][0]]
            y_list = [self.xt[:, indx - 1][1], self.xt[:, indx][1]]
            self.ax.plot(x_list, y_list, '-o', color='r')

            # show past trajecotry of each dynamic obstacle
            if self.show_obs:
                if self.cir_obs_list_t is not None:
                    for i in range(self.cir_obs_num):
                        ox_list = [self.cir_obs_list_t[i][:, indx - 1][0], self.cir_obs_list_t[i][:, indx][0]]
                        oy_list = [self.cir_obs_list_t[i][:, indx - 1][1], self.cir_obs_list_t[i][:, indx][1]]
                        self.ax.plot(ox_list, oy_list, linestyle='--', color='k', )

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
