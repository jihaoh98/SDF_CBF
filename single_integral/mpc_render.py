import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np


class Mpc_Render:
    def __init__(self, robot_params, obs_params, dt) -> None:
        """ Visulization for MPC-DCBF """
        self.dt = dt

        self.radius = robot_params["radius"]
        self.vx_min = robot_params["vx_boundary"][0]
        self.vx_max = robot_params["vx_boundary"][1]
        self.vy_min = robot_params["vy_boundary"][0]
        self.vy_max = robot_params["vy_boundary"][1]

        self.obs_num = len(obs_params["obs_states"])
        
        self.predict_states = None
        self.current_states = None
        # in shape (obs_num, time step, 5)
        self.obs_list = None

        # plot
        self.fig, self.ax = plt.subplots()

        self.start_body = None
        self.target_body = None
        self.robot_body = None
        self.predict_line = None
        
        self.obs = [None for i in range(self.obs_num)]
        
        # settings of Times New Roman
        # set the text in Times New Roman
        config = {
            "font.family": "serif",
            "font.size": 12,
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
        }
        plt.rcParams.update(config)

        # set the label in Times New Roman and size
        self.label_font = {
            "family": "Times New Roman",
            "weight": "normal",
            "size": 25, 
        }

        # set the legend font
        self.legend_font = {"family": "Times New Roman", "weight": "normal", "size": 25}  

        self.fig.set_size_inches(7.0, 7.0)
        self.fig.set_dpi(150)
        self.ax.set_aspect("equal")

        self.ax.set_xlim(-0.1, 6.0)
        self.ax.set_ylim(-0.1, 6.0)

        self.ax.set_xlabel("x (m)", self.label_font)
        self.ax.set_ylabel("y (m)", self.label_font)
        self.ax.set_title("MPC-DCBF", self.label_font)

        # set the tick in Times New Roman and size
        self.ax.tick_params(labelsize=25)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname("Times New Roman") for label in labels]

    def cir_render(self, x, x_predict, target, obs_states):
        """ 
        Visulization 
        Args:
            x: actual states in each steps in shape (time_step, 2) list 
            x_predict: predict_states in each steps
            target: target state of robot
            obs_states: states of obstacles in each steps in shape (obs_num, time step, 5)
        """
        self.current_states = np.array(x).reshape(-1, 2)
        self.predict_states = x_predict
        self.obs_list = np.array(obs_states)

        start_position = self.current_states[0]
        self.cir_animation_init(start_position, target)

        self.robot_body = mpatches.Circle(
            xy=(start_position[0], start_position[1]),
            radius=self.radius,
            color="red",
            fill=False,
            lw=2,
        )
        self.ax.add_patch(self.robot_body)

        for i in range(self.obs_num):
            self.obs[i] = mpatches.Circle(
                xy=(self.obs_list[i][0, 0], self.obs_list[i][0, 1]),
                radius=self.obs_list[i][0, 4],
                color="k",
                fill=False,
                lw=2,
                zorder=5
            )
            self.ax.add_patch(self.obs[i])

        self.ani = animation.FuncAnimation(
            self.fig, 
            func=self.cir_animation_loop, 
            frames=len(x_predict), 
            interval=200, 
            repeat=False,
        )

        plt.grid("--")
        # writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        # self.ani.save('mpc_dcbf.gif', writer=writer)
        plt.show()

    def cir_animation_init(self, start, target):
        """ animation init """
        self.start_body, = plt.plot(start[0], start[1], color='limegreen', marker="*", markersize=12)
        self.target_body, = plt.plot(target[0], target[1], color='purple', marker="*", markersize=12)

        return self.ax.patches + self.ax.texts + self.ax.artists
    
    def cir_animation_loop(self, indx):
        """ loop for update the state of robot """
        self.robot_body.remove()
        for i in range(self.obs_num):
            self.obs[i].remove()
        if self.predict_line is not None:
            self.predict_line.remove()

        self.robot_body = mpatches.Circle(
            xy=(self.current_states[indx, 0], self.current_states[indx, 1]),
            radius=self.radius,
            color="red",
            fill=False,
            lw=2,
        )
        self.ax.add_patch(self.robot_body)

        self.predict_line, = self.ax.plot(
            self.predict_states[indx][:, 0], 
            self.predict_states[indx][:, 1],
            color = 'grey',
            linewidth = 3,
        )
        # dynamic obstacle
        for i in range(self.obs_num):
            self.obs[i] = mpatches.Circle(
                xy=(self.obs_list[i][indx, 0], self.obs_list[i][indx, 1]),
                radius=self.obs_list[i][0, 4],
                color="k",
                fill=False,
                lw=2,
                zorder=5
            )
            self.ax.add_patch(self.obs[i])

        # trajectory
        if indx != 0:
            x_list = [self.current_states[indx - 1][0], self.current_states[indx][0]]
            y_list = [self.current_states[indx - 1][1], self.current_states[indx][1]]
            self.ax.plot(x_list, y_list, color="b")

            for i in range(self.obs_num):
                ox_list = [self.obs_list[i][indx - 1, 0], self.obs_list[i][indx, 0]]
                oy_list = [self.obs_list[i][indx - 1, 1], self.obs_list[i][indx, 1]]
                self.ax.plot(ox_list, oy_list, linestyle='--', color='k',)

        # plt.savefig('/home/hjh/ALSPG/alspg/vo_alspg/figure/{}.png'.format(indx), format='png', dpi=300)
        return self.ax.patches + self.ax.texts + self.ax.artists
    