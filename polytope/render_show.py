import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import numpy as np


class Render_Animation:
    def __init__(self, robot, robot_params, obs_list, cir_obs_params, dt) -> None:
        """ init the render animation """
        self.dt = dt

        # robot
        self.robot = robot
        self.robot_model = self.robot.model
        self.robot_init_state = robot.init_state 
        self.robot_target_state = np.array(robot_params['target_state'])
        self.robot_width = robot_params['width']
        self.robot_height = robot_params['height']
        self.umax = robot_params['u_max']
        self.umin = robot_params['u_min']

        # obstacle
        self.obs_list = obs_list
        if obs_list is not None:
            self.obs_num = len(obs_list)
            self.obs = [None for i in range(self.obs_num)]

        if cir_obs_params is not None:
            self.cir_obs_num = len(cir_obs_params['obs_states'])
            self.cir_obs = [None for i in range(self.cir_obs_num)]

        # storage the past states of robots and different shaped obs
        self.xt = None
        self.obs_list_t = None
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

        # label_font
        self.label_font = {'family': 'Times New Roman', 
                           'weight': 'normal', 
                           'size': 16,
                           }
        
        # legend font
        self.legend_font = {"family": "Times New Roman", "weight": "normal", "size": 12}
        
    def render(self, xt, obs_list_t, cir_obs_list_t, terminal_time, show_obs, save_gif=False):
        """ Visualization """
        self.fig.set_size_inches(7, 6.5)
        self.fig.set_dpi(150)
        self.ax.set_aspect('equal')

        self.ax.set_xlim(-1, 15.0)
        self.ax.set_ylim(-1, 15.0)

        self.ax.set_xlabel('x (m)', self.label_font)
        self.ax.set_ylabel("y (m)", self.label_font)

        # set the tick in Times New Roman and size
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        self.xt = xt
        self.obs_list_t = obs_list_t
        self.cir_obs_list_t = cir_obs_list_t
        self.show_obs = show_obs

        self.animation_init()

        # robot and the arrow
        init_vertexes = self.robot.get_vertexes(self.robot_init_state)
        self.robot_body = mpatches.Polygon(init_vertexes, edgecolor='silver', facecolor=None)
        self.ax.add_patch(self.robot_body)

        if self.robot_model == 'unicycle':
            self.robot_arrow = mpatches.Arrow(
                self.robot_init_state[0],
                self.robot_init_state[1],
                self.robot_width * np.cos(self.robot_init_state[2]),
                self.robot_width * np.sin(self.robot_init_state[2]),
                width=0.15,
                color='k',
            )
            self.ax.add_patch(self.robot_arrow)

        # show obstacles
        if self.show_obs:
            if self.obs_list_t is not None:
                for i in range(self.obs_num):
                    obs_vertexes = self.obs_list[i].get_current_vertexes(self.obs_list_t[i][:, 0])
                    self.obs[i] = mpatches.Polygon(obs_vertexes, color='k')
                    self.ax.add_patch(self.obs[i]) 
            if self.cir_obs_list_t is not None:
                for i in range(self.cir_obs_num):
                    self.cir_obs[i] = mpatches.Circle(xy=self.cir_obs_list_t[i][0:2, 0], radius=self.cir_obs_list_t[i][4, 0], color='k')
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
        # start body and arrow
        start_vertexes = self.robot.get_vertexes(self.robot_init_state)
        self.start_body = mpatches.Polygon(start_vertexes, edgecolor='silver', facecolor=None)
        self.ax.add_patch(self.start_body)
        self.start_body.set_zorder(0)

        if self.robot_model == 'unicycle':
            self.start_arrow = mpatches.Arrow(
                self.robot_init_state[0],
                self.robot_init_state[1], 
                self.robot_width * np.cos(self.robot_init_state[2]),
                self.robot_width * np.sin(self.robot_init_state[2]),
                width=0.15,
                color='k',
            )
            self.ax.add_patch(self.start_arrow)
            self.start_arrow.set_zorder(1)

        # target position 
        self.end_body = mpatches.Circle((self.robot_target_state[0], self.robot_target_state[1]), radius=0.5, color='silver')
        self.ax.add_patch(self.end_body)
        self.end_body.set_zorder(0)
    
        return self.ax.patches + self.ax.texts + self.ax.artists
    
    def animation_loop(self, indx):
        """ loop for update the position of robot and obstacles """
        # robot
        self.robot_body.remove()
        cur_vertexes = self.robot.get_vertexes(self.xt[:, indx])
        self.robot_body = mpatches.Polygon(cur_vertexes, edgecolor='r', facecolor=None)
        self.ax.add_patch(self.robot_body)

        if self.robot_model == 'unicycle':
            self.robot_arrow.remove()
            self.robot_arrow = mpatches.Arrow(
                self.xt[:, indx][0],
                self.xt[:, indx][1],
                self.robot_width * np.cos(self.xt[:, indx][2]),
                self.robot_width * np.sin(self.xt[:, indx][2]),
                width=0.15, 
                color='k',
            )
            self.ax.add_patch(self.robot_arrow)

        # obs
        if self.show_obs:
            if self.obs_list_t is not None:
                for i in range(self.obs_num):
                    self.obs[i].remove()
                    obs_vertexes = self.obs_list[i].get_current_vertexes(self.obs_list_t[i][:, indx])
                    self.obs[i] = mpatches.Polygon(obs_vertexes, color='k')
                    self.ax.add_patch(self.obs[i]) 

            if self.cir_obs_list_t is not None:
                for i in range(self.cir_obs_num):
                    self.cir_obs[i].remove()
                    self.cir_obs[i] = mpatches.Circle(xy=self.cir_obs_list_t[i][:, indx][0:2], 
                                                      radius=self.cir_obs_list_t[i][:, indx][4], color='k')
                    self.ax.add_patch(self.cir_obs[i]) 

        # show past trajecotry of robot and obstacles
        if indx != 0:
            x_list = [self.xt[:, indx - 1][0], self.xt[:, indx][0]]
            y_list = [self.xt[:, indx - 1][1], self.xt[:, indx][1]]
            self.ax.plot(x_list, y_list, color='b',)

            # show past trajecotry of each dynamic obstacle
            if self.show_obs:
                if self.obs_list_t is not None:
                    for i in range(self.obs_num):
                        ox_list = [self.obs_list_t[i][:, indx - 1][0], self.obs_list_t[i][:, indx][0]]
                        oy_list = [self.obs_list_t[i][:, indx - 1][1], self.obs_list_t[i][:, indx][1]]  
                        self.ax.plot(ox_list, oy_list, linestyle='--', color='k',)
                if self.cir_obs_list_t is not None:
                    for i in range(self.cir_obs_num):
                        ox_list = [self.cir_obs_list_t[i][:, indx - 1][0], self.cir_obs_list_t[i][:, indx][0]]
                        oy_list = [self.cir_obs_list_t[i][:, indx - 1][1], self.cir_obs_list_t[i][:, indx][1]]  
                        self.ax.plot(ox_list, oy_list, linestyle='--', color='k',)

        # plt.savefig('figure/{}.png'.format(indx), format='png', dpi=300)
        return self.ax.patches + self.ax.texts + self.ax.artists
    
    def show_clf(self, clf_type, clft, terminal_time):
        """ show changes in clf """
        t = np.arange(0, terminal_time * self.dt, self.dt)[0:terminal_time]
        
        # add dpi
        figure = plt.figure()
        figure.set_dpi(150)

        plt.plot(t, clft[0:terminal_time].reshape(terminal_time, ), linewidth=3, color='red')
        plt.title('Changes in CLF of {}'.format(clf_type), self.label_font)
        plt.ylabel('V(x)', self.label_font)
        plt.xlabel('Time (s)', self.label_font)

        # set the tick in Times New Roman and size
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        plt.grid()
        # plt.savefig('clf_{}.png'.format(clf_type), format='png', dpi=300)
        plt.show()

    def show_slack(self, clf_type, slackt, terminal_time):
        """ show changes in clf """
        t = np.arange(0, terminal_time * self.dt, self.dt)[0:terminal_time]
        
        # add dpi
        figure = plt.figure()
        figure.set_dpi(150)

        plt.plot(t, slackt[0:terminal_time].reshape(terminal_time, ), linewidth=3, color='blue')
        plt.title('Changes in slack variable of {}'.format(clf_type), self.label_font)
        plt.ylabel('Slack Variable', self.label_font)
        plt.xlabel('Time (s)', self.label_font)

        # set the tick in Times New Roman and size
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        plt.grid()
        # plt.savefig('slack_{}.png'.format(clf_type), format='png', dpi=300)
        plt.show()

    def show_cbf(self, i, cbft, terminal_time):
        """ show changes in cbf """
        t = np.arange(0, terminal_time * self.dt, self.dt)[0:terminal_time]
        
        # add dpi
        figure = plt.figure()
        figure.set_dpi(150)

        plt.plot(t, cbft[i, 0:terminal_time].reshape(terminal_time, ), linewidth=3, color='blue')
        plt.title('SDF with respect to {}th obstacle'.format(i), self.label_font)
        plt.ylabel('cbf (m)', self.label_font)
        plt.xlabel('Time (s)', self.label_font)

        # set the tick in Times New Roman and size
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        plt.grid() 
        # plt.savefig('cbf.png', format='png', dpi=300)
        plt.show()

    def show_both_cbf(self, i, cbft, cir_cbft, terminal_time):
        """ show both cbf """
        t = np.arange(0, terminal_time * self.dt, self.dt)[0:terminal_time]
        
        # add dpi
        figure = plt.figure()
        figure.set_dpi(150)

        plt.plot(t, cir_cbft[i, 0:terminal_time].reshape(terminal_time), linewidth=3, color='grey', label='circular-shaped obstacle')
        plt.plot(t, cbft[i, 0:terminal_time].reshape(terminal_time), linewidth=3, color='blue', linestyle='dashed', label='polytopic-shaped obstacle')
        plt.title('RC-ESDF (CBF) with respect to L-shaped robot', self.label_font)
        plt.ylabel('distance (m)', self.label_font)
        plt.xlabel('Time (s)', self.label_font)

        # set the tick in Times New Roman and size
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        plt.legend(loc="upper right", prop=self.legend_font)
        plt.grid() 
        # plt.savefig('both_cbf.png', format='png', dpi=300)
        plt.show()

    def show_integral_controls(self, ut, terminal_time):
        """ show controls of integral model """
        t = np.arange(0, terminal_time * self.dt, self.dt)[0:terminal_time]

        # add dpi
        figure = plt.figure()
        figure.set_dpi(150)

        plt.plot(t, ut[0][0:terminal_time].reshape(terminal_time,), linewidth=3, color="b", label="vx",)
        plt.plot(t, self.umax[0] * np.ones(t.shape[0]), color="b", linestyle='dashed') 
        plt.plot(t, self.umin[0] * np.ones(t.shape[0]), color="b", linestyle='dashed')

        plt.plot(t, ut[1][0:terminal_time].reshape(terminal_time,),linewidth=3, color="r", label="vy",)
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

    def show_unicycle_model_controls(self, ut, terminal_time):
        t = np.arange(0, terminal_time * self.dt, self.dt)[0:terminal_time]

        # add dpi
        figure = plt.figure()
        figure.set_dpi(150)

        plt.plot(t, ut[0][0:terminal_time].reshape(terminal_time,), linewidth=3, color="b", label="v",)
        plt.plot(t, self.umax[0] * np.ones(t.shape[0]), color="b", linestyle='dashed') 
        plt.plot(t, self.umin[0] * np.ones(t.shape[0]), color="b", linestyle='dashed')

        plt.plot(t, ut[1][0:terminal_time].reshape(terminal_time,),linewidth=3, color="r", label="w",)
        plt.plot(t, self.umax[1] * np.ones(t.shape[0]), color="r", linestyle='dashed') 
        plt.plot(t, self.umin[1] * np.ones(t.shape[0]), color="r", linestyle='dashed')

        plt.title("Control Variables", self.label_font)
        plt.xlabel("Time (s)", self.label_font)
        plt.ylabel("v (m/s) / w (rad/s)", self.label_font)

        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname("Times New Roman") for label in labels]

        plt.legend(loc="upper right", prop=self.legend_font)
        plt.grid()
        # plt.savefig("controls.png", format="png", dpi=300)
        plt.show()


















