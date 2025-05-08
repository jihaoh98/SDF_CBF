import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.ticker import FormatStrFormatter
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
        # self.robot_width = self.robot.width
        # self.robot_height = self.robot.height
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
        self.robot_body_A = None
        self.robot_body_B = None
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
        self.label_font = {
            'family': 'Times New Roman', 
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
        init_vertexes = self.robot.get_vertices(self.robot_init_state)
        self.robot_body_A = mpatches.Polygon(init_vertexes[0], alpha=0.5, color='red')
        self.ax.add_patch(self.robot_body_A)
        self.robot_body_B = mpatches.Polygon(init_vertexes[1], alpha=0.5, color='blue')
        self.ax.add_patch(self.robot_body_B)

        if self.robot_model == 'unicycle':
            self.robot_arrow = mpatches.Arrow(
                self.robot_init_state[0],
                self.robot_init_state[1],
                0.1 / 2 * np.cos(self.robot_init_state[2]),
                0.1 / 2 * np.sin(self.robot_init_state[2]),
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
        start_vertexes = self.robot.get_vertices(self.robot_init_state)
        self.robot_body_A = mpatches.Polygon(start_vertexes[0], alpha=0.5, color='red')
        self.ax.add_patch(self.robot_body_A)
        self.robot_body_B = mpatches.Polygon(start_vertexes[1], alpha=0.5, color='blue')
        self.ax.add_patch(self.robot_body_B)

        # self.robot_body_A.set_zorder(0)
        # self.robot_body_B.set_zorder(1)

        if self.robot_model == 'unicycle':
            self.start_arrow = mpatches.Arrow(
                self.robot_init_state[0],
                self.robot_init_state[1], 
                0.1 / 2 * np.cos(self.robot_init_state[2]),
                0.1 / 2 * np.sin(self.robot_init_state[2]),
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
        # self.robot_body_A.remove()
        # self.robot_body_B.remove()

        cur_vertexes = self.robot.get_vertices(self.xt[:, indx])
        self.robot_body_A = mpatches.Polygon(cur_vertexes[0], alpha=0.5, color='red')
        self.ax.add_patch(self.robot_body_A)
        self.robot_body_B = mpatches.Polygon(cur_vertexes[1], alpha=0.5, color='blue')
        self.ax.add_patch(self.robot_body_B)

        if self.robot_model == 'unicycle':
            self.robot_arrow.remove()
            self.robot_arrow = mpatches.Arrow(
                self.xt[:, indx][0],
                self.xt[:, indx][1],
                0.1 / 2 * np.cos(self.xt[:, indx][2]),
                0.1 / 2 * np.sin(self.xt[:, indx][2]),
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

    def show_integral_cbf(self, cbft, cir_cbft, terminal_time):
        """ show both cbf """
        figure, ax = plt.subplots(figsize=(16, 9))
        figure.set_dpi(200)

        font_path = "/home/hjh/simfang.ttf"  
        legend_font = fm.FontProperties(fname=font_path, size=35)
        label_font = fm.FontProperties(fname=font_path, size=35)
        color_list = ['#219EBC', '#FEB705']

        t = np.arange(0, (terminal_time) * self.dt, self.dt)[0:terminal_time]
        cbf, = plt.plot(
            t, cbft[0, 0:terminal_time].reshape(terminal_time), 
            linewidth=6, color=color_list[0]
        )
        cir_cbf, = plt.plot(
            t, cir_cbft[0, 0:terminal_time].reshape(terminal_time),
            linewidth=6, color=color_list[1]
        )

        plt.xlabel("时间" + r'$(s)$', fontproperties=label_font)
        plt.ylabel("距离" + r'$(m)$', fontproperties=label_font)
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.legend(
            handles=[cbf, cir_cbf], 
            labels=['多边形障碍物', '圆形障碍物'], 
            loc='lower right', prop=legend_font
        )

        # set the tick in Times New Roman and size
        ax.tick_params(labelsize=40)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        plt.grid() 
        plt.savefig('integral_cbf.png', format='png', dpi=300, bbox_inches='tight')

    def show_integral_controls(self, ut, terminal_time):
        """ show controls of integral model """
        figure, ax = plt.subplots(figsize=(16, 9))
        figure.set_dpi(200)
        font_path = "/home/hjh/simfang.ttf"  
        legend_font = {"family": "Times New Roman", "weight": "normal", "size": 35}
        label_font = fm.FontProperties(fname=font_path, size=35)

        t = np.arange(0, terminal_time * self.dt, self.dt)[0:terminal_time]
        v_x = ut[0][0:terminal_time].reshape(terminal_time,)
        v_y = ut[1][0:terminal_time].reshape(terminal_time,)
        # vx, = plt.plot(t, v_x, linewidth=6, color="#FFDF92")
        # vy, = plt.plot(t, v_y, linewidth=6, color="#90BEE0")

        window_size = 5
        vx_smooth = np.convolve(v_x, np.ones(window_size) / window_size, mode='valid')
        vy_smooth = np.convolve(v_y, np.ones(window_size) / window_size, mode='valid')
        vx, = plt.plot(t[:vx_smooth.size], vx_smooth, linewidth=6, color="#FFDF92")
        vy, = plt.plot(t[:vy_smooth.size], vy_smooth, linewidth=6, color="#90BEE0")

        # add boundary
        v_min, = plt.plot(
            t, self.umin[0] * np.ones(t.shape[0]),
            linewidth=6, color="#DB3124", linestyle="--"
        )
        v_max, = plt.plot(
            t, self.umax[0] * np.ones(t.shape[0]),
            linewidth=6, color="#4B74B2", linestyle="--"
        )

        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.xlabel("时间" + r'$(s)$', fontproperties=label_font)
        plt.ylabel('速度' + r'$(m/s)$', fontproperties=label_font)
        
        plt.legend(
            handles=[vx, vy, v_min, v_max], 
            labels=[r'$v_x$', r'$v_y$', r'$v_{min}$', r'$v_{max}$'], 
            loc='lower right', prop=legend_font,
            framealpha=0.5, ncol=4,
            bbox_to_anchor=(0.95, 0.05)
        )

        # set the tick in Times New Roman and size
        ax.tick_params(labelsize=45)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        plt.grid() 
        plt.savefig('controls_integral.png', format='png', dpi=300, bbox_inches='tight')
    
    def show_unicycle_cbf(self, cbft, terminal_time):
        figure, ax = plt.subplots(figsize=(16, 9))
        figure.set_dpi(200)

        font_path = "/home/hjh/simfang.ttf"  
        legend_font = fm.FontProperties(fname=font_path, size=35)
        label_font = fm.FontProperties(fname=font_path, size=35)
        color_list = ['#219EBC', '#FEB705']

        t = np.arange(0, (terminal_time) * self.dt, self.dt)[0:terminal_time]
        cbf1, = plt.plot(
            t, cbft[0, 0:terminal_time].reshape(terminal_time), 
            linewidth=6, color=color_list[0]
        )
        cbf2, = plt.plot(
            t, cbft[1, 0:terminal_time].reshape(terminal_time), 
            linewidth=6, color=color_list[1]
        )

        plt.xlabel("时间" + r'$(s)$', fontproperties=label_font)
        plt.ylabel("距离" + r'$(m)$', fontproperties=label_font)
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.legend(
            handles=[cbf1, cbf2], 
            labels=['多边形障碍物1', '多边形障碍物2'], 
            loc='upper left', prop=legend_font
        )

        # set the tick in Times New Roman and size
        ax.tick_params(labelsize=40)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        plt.grid() 
        plt.savefig('unicycle_cbf.png', format='png', dpi=300, bbox_inches='tight')

    def show_unicycle_model_controls(self, ut, terminal_time):
        """ show controls of unicycle model with dual y-axis """
        figure, ax1 = plt.subplots(figsize=(16, 9))
        figure.set_dpi(200)
        font_path = "/home/hjh/simfang.ttf"  
        legend_font = {"family": "Times New Roman", "weight": "normal", "size": 25}
        label_font = fm.FontProperties(fname=font_path, size=35)

        v_color = ['#EDDDC3', '#8EB69C', '#4EAB90']
        w_color = ['#EEBF6D', '#D94F33', '#834026']

        t = np.arange(0, terminal_time * self.dt, self.dt)[0:terminal_time]
        v = ut[0][0:terminal_time].reshape(terminal_time,)
        w = ut[1][0:terminal_time].reshape(terminal_time,)

        window_size = 5
        v_smooth = np.convolve(v, np.ones(window_size) / window_size, mode='valid')
        w_smooth = np.convolve(w, np.ones(window_size) / window_size, mode='valid')

        ax1.set_xlabel("时间" + r'$(s)$', fontproperties=label_font)
        ax1.set_ylabel("线速度" + r'$v (m/s)$', fontproperties=label_font)
        ax1.tick_params(axis='y')

        vv, = ax1.plot(t[:v_smooth.size], v_smooth, linewidth=6, color=v_color[0])
        v_min, = ax1.plot(t, self.umin[0] * np.ones(t.shape[0]), linewidth=6, color=v_color[1], linestyle="--")
        v_max, = ax1.plot(t, self.umax[0] * np.ones(t.shape[0]), linewidth=6, color=v_color[2], linestyle="--")

        ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax1.grid(True, linestyle='--', alpha=0.6)  # 优化网格

        ax2 = ax1.twinx()
        ax2.set_ylim(ax1.get_ylim())  
        ax2.set_ylabel("角速度" + r'$w (rad/s)$', fontproperties=label_font)
        ax2.tick_params(axis='y')
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ww, = ax2.plot(t[:w_smooth.size], w_smooth, linewidth=6, color=w_color[0])
        w_min, = ax2.plot(t, self.umin[1] * np.ones(t.shape[0]), linewidth=6, color=w_color[1], linestyle="--")
        w_max, = ax2.plot(t, self.umax[1] * np.ones(t.shape[0]), linewidth=6, color=w_color[2], linestyle="--")

        lines = [vv, ww, v_min, v_max, w_min, w_max]
        labels = [r'$v$', r'$w$', r'$v_{min}$', r'$v_{max}$', r'$w_{min}$', r'$w_{max}$']
        ax1.legend(lines, labels, loc='lower center', prop=legend_font, framealpha=0.5, ncol=6, bbox_to_anchor=(0.5, 0.05))

        ax1.tick_params(labelsize=45)
        ax2.tick_params(labelsize=45)

        # **保存图片**
        plt.savefig('controls_unicycle.png', format='png', dpi=300, bbox_inches='tight')

    def show_integral_model(self, xt, obs_list_t, cir_obs_list_t, terminal_time, index_t):
        font_path = "/home/hjh/simfang.ttf"  
        fangsong_font = fm.FontProperties(fname=font_path, size=20)

        label_font = {
            'family': 'Times New Roman', 
            'weight': 'normal', 
            'size': 30,
        }

        figure, ax = plt.subplots(figsize=(10, 9))
        figure.set_dpi(300)
        ax.set_aspect('equal')
        ax.set_xlim(-1.0, 15.0)
        ax.set_ylim(-1.0, 15.0)

        plt.xlabel("x (m)", label_font)
        plt.ylabel("y (m)", label_font)
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax.tick_params(labelsize=35)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        start_color = ['#AAD2E6', '#90BEE0', '#4B74B2', '#3C6478']
        trajecotry_color = ['#4B74B2']

        # start body, arrow and target position
        start_vertexes = self.robot.get_vertexes(self.robot_init_state)
        start_body = mpatches.Polygon(
            start_vertexes, 
            edgecolor=start_color[1], 
            facecolor=start_color[0]
        )
        ax.add_patch(start_body)
        start_body.set_zorder(1)

        plt.plot(self.robot_target_state[0], self.robot_target_state[1], color='purple', marker="*", markersize=25, zorder=1)

        # trajectory of the robot
        tra_x = xt[0, 0:terminal_time + 1]
        tra_y = xt[1, 0:terminal_time + 1]
        plt.plot(tra_x, tra_y, color=trajecotry_color[0], linewidth=5, zorder=0)

        for i in range(self.obs_num):
            plt.plot(obs_list_t[i][0, 0], obs_list_t[i][1, 0], color='k', marker="s", markersize=10, zorder=0)
            plt.plot(
                obs_list_t[i][0, :terminal_time + 1], obs_list_t[i][1, :terminal_time + 1], 
                color='k', linewidth=5, linestyle='--', zorder=1
            )
        
        start_proxy = plt.scatter([], [], s=500, edgecolor=start_color[1], facecolor=start_color[0], linewidths=2)
        tar_proxy = plt.scatter([], [], s=500, edgecolor='purple', facecolor='purple', marker='*')
        obs_proxy = plt.scatter([], [], s=500, edgecolor='k', linestyle='--', facecolor='none', linewidths=1.5)
        obs_start_proxy = plt.scatter([], [], s=300, edgecolor='k', facecolor='k', marker='s')
        plt.legend(
            handles=[start_proxy, tar_proxy, obs_proxy, obs_start_proxy], 
            labels=['起点', '目标点', '障碍物', '障碍物起点'], loc='upper left',
            prop=fangsong_font
        )

        robot = [None for i in range(len(index_t))]
        obstacle = [[None for j in range(len(index_t))] for i in range(self.obs_num)]
        alpha_index = [0.15, 0.5, 0.85]

        for i in range(len(index_t)):
            robot[i] = mpatches.Polygon(
                self.robot.get_vertexes(xt[:, index_t[i]]), 
                edgecolor=start_color[3], 
                fill=False,
                lw=4,
                alpha=alpha_index[i]
            )
            ax.add_patch(robot[i])
            robot[i].set_zorder(1)

            for j in range(self.obs_num):
                obstacle[j][i] = mpatches.Polygon(
                    self.obs_list[j].get_current_vertexes(obs_list_t[j][:, index_t[i]]), 
                    edgecolor='k', 
                    fill=False,
                    lw=4,
                    linestyle='--',
                    alpha=alpha_index[i],
                    zorder=0
                )
                ax.add_patch(obstacle[j][i])

        # final position
        end_vertexes = self.robot.get_vertexes(xt[:, terminal_time])
        end_body = mpatches.Polygon(
            end_vertexes, 
            edgecolor=start_color[3], 
            fill=False,
            lw=4
        )
        ax.add_patch(end_body)
        end_body.set_zorder(1)

        obs_final = [None for i in range(self.obs_num)]
        for i in range(self.obs_num):
            obs_final[i] = mpatches.Polygon(
                self.obs_list[i].get_current_vertexes(obs_list_t[i][:, terminal_time]), 
                edgecolor='k', 
                fill=False,
                lw=4,
                linestyle='--',
                zorder=0
            )
            ax.add_patch(obs_final[i])

        cir_obs = [None for i in range(self.cir_obs_num)]
        for i in range(self.cir_obs_num):
            cir_obs[i] = mpatches.Circle(
                xy=cir_obs_list_t[i][:, terminal_time][0:2], 
                radius=cir_obs_list_t[i][:, terminal_time][4], 
                color='k',
                lw=4,
                linestyle='--',
                fill=False,
                zorder=0
            )
            ax.add_patch(cir_obs[i]) 

        plt.savefig('integral.png', format='png', dpi=300, bbox_inches='tight')
       
    def show_unicycle_model(self, xt, obs_list_t, terminal_time, index_t):
        font_path = "/home/hjh/simfang.ttf"  
        fangsong_font = fm.FontProperties(fname=font_path, size=20)

        label_font = {
            'family': 'Times New Roman', 
            'weight': 'normal', 
            'size': 30,
        }

        figure, ax = plt.subplots(figsize=(10, 9))
        figure.set_dpi(300)
        ax.set_aspect('equal')
        ax.set_xlim(-1.0, 15.0)
        ax.set_ylim(-1.0, 15.0)

        plt.xlabel("x (m)", label_font)
        plt.ylabel("y (m)", label_font)
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax.tick_params(labelsize=35)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        start_color = ['#AAD2E6', '#90BEE0', '#4B74B2', '#3C6478']
        trajecotry_color = ['#4B74B2']

        # start body, arrow and target position
        start_vertexes = self.robot.get_vertexes(self.robot_init_state)
        start_body = mpatches.Polygon(
            start_vertexes, 
            edgecolor=start_color[1], 
            facecolor=start_color[0]
        )
        ax.add_patch(start_body)
        start_body.set_zorder(1)

        start_arrow = mpatches.Arrow(
            self.robot_init_state[0],
            self.robot_init_state[1], 
            self.robot_width / 2 * np.cos(self.robot_init_state[2]),
            self.robot_width / 2 * np.sin(self.robot_init_state[2]),
            width=0.2,
            color='k',
        )
        ax.add_patch(start_arrow)
        start_arrow.set_zorder(2)
        plt.plot(self.robot_target_state[0], self.robot_target_state[1], color='purple', marker="*", markersize=25, zorder=1)

        # trajectory of the robot
        tra_x = xt[0, 0:terminal_time + 1]
        tra_y = xt[1, 0:terminal_time + 1]
        plt.plot(tra_x, tra_y, color=trajecotry_color[0], linewidth=5, zorder=0)

        # start position of obstacle and trajectory
        for i in range(self.obs_num):
            plt.plot(obs_list_t[i][0, 0], obs_list_t[i][1, 0], color='k', marker="s", markersize=10, zorder=0)
            plt.plot(
                obs_list_t[i][0, :terminal_time + 1], obs_list_t[i][1, :terminal_time + 1], 
                color='k', linewidth=5, linestyle='--', zorder=1
            )

        start_proxy = plt.scatter([], [], s=500, edgecolor=start_color[1], facecolor=start_color[0], linewidths=2)
        tar_proxy = plt.scatter([], [], s=500, edgecolor='purple', facecolor='purple', marker='*')
        obs_proxy = plt.scatter([], [], s=500, edgecolor='k', linestyle='--', facecolor='none', linewidths=1.5)
        obs_start_proxy = plt.scatter([], [], s=300, edgecolor='k', facecolor='k', marker='s')
        plt.legend(
            handles=[start_proxy, tar_proxy, obs_proxy, obs_start_proxy], 
            labels=['起点', '目标点', '障碍物', '障碍物起点'], loc='upper left',
            prop=fangsong_font
        )

        # dynamic obstacle and robot
        robot = [None for i in range(len(index_t))]
        robot_arrow = [None for i in range(len(index_t))]
        obstacle = [[None for j in range(len(index_t))] for i in range(self.obs_num)]
        alpha_index = [0.2, 0.6]

        for i in range(len(index_t)):
            robot[i] = mpatches.Polygon(
                self.robot.get_vertexes(xt[:, index_t[i]]), 
                edgecolor=start_color[3], 
                fill=False,
                lw=4,
                alpha=alpha_index[i]
            )
            ax.add_patch(robot[i])
            robot[i].set_zorder(1)

            robot_arrow[i] = mpatches.Arrow(
                xt[0, index_t[i]], 
                xt[1, index_t[i]], 
                self.robot_width / 2 * np.cos(xt[2, index_t[i]]), 
                self.robot_width / 2 * np.sin(xt[2, index_t[i]]),
                width=0.2,
                color='k',
                alpha=alpha_index[i]
            )
            ax.add_patch(robot_arrow[i])
            robot_arrow[i].set_zorder(2)

            for j in range(self.obs_num):
                obstacle[j][i] = mpatches.Polygon(
                    self.obs_list[j].get_current_vertexes(obs_list_t[j][:, index_t[i]]), 
                    edgecolor='k', 
                    fill=False,
                    lw=4,
                    linestyle='--',
                    alpha=alpha_index[i],
                    zorder=0
                )
                ax.add_patch(obstacle[j][i])

        # final position
        end_vertexes = self.robot.get_vertexes(xt[:, terminal_time])
        end_body = mpatches.Polygon(
            end_vertexes, 
            edgecolor=start_color[3], 
            fill=False,
            lw=4
        )
        ax.add_patch(end_body)
        end_body.set_zorder(1)

        end_arrow = mpatches.Arrow(
            xt[0, terminal_time], 
            xt[1, terminal_time], 
            self.robot_width / 2 * np.cos(xt[2, terminal_time]), 
            self.robot_width / 2 * np.sin(xt[2, terminal_time]),
            width=0.2,
            color='k',
        )
        ax.add_patch(end_arrow)
        end_arrow.set_zorder(2)

        obs_final = [None for i in range(self.obs_num)]
        for i in range(self.obs_num):
            obs_final[i] = mpatches.Polygon(
                self.obs_list[i].get_current_vertexes(obs_list_t[i][:, terminal_time]), 
                edgecolor='k', 
                fill=False,
                lw=4,
                linestyle='--',
                zorder=0
            )
            ax.add_patch(obs_final[i])

        plt.savefig('unicycle.png', format='png', dpi=300, bbox_inches='tight')



