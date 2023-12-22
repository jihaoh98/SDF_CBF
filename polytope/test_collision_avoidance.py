import numpy as np
import sdf_qp
import obs
import polytopic_robot
import time
import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches


class Collision_Avoidance:
    def __init__(self) -> None:
        """collision avoidance with obstacles"""
        file_name = "./settings.yaml"
        with open(file_name) as file:
            config = yaml.safe_load(file)

        robot_params = config["robot"]
        controller_params = config["controller"]
        self.cbf_qp = sdf_qp.Sdf_Cbf_Clf()

        # if no data, return N
        obs_params = config.get("obstacle_list")
        cir_obs_params = config.get("cir_obstacle_list")

        # initialize the robot state, half width and height
        self.robot_width = robot_params["width"]
        self.robot_height = robot_params["height"]
        self.u_min = robot_params["u_min"]
        self.u_max = robot_params["u_max"]
        init_state = np.array(robot_params["initial_state"])
        # in shape (N, 2)
        self.robot_vertexes = np.array(
            [
                [init_state[0] - self.robot_width, init_state[1] - self.robot_height],
                [init_state[0] + self.robot_width, init_state[1] - self.robot_height],
                [init_state[0] + self.robot_width, init_state[1] + self.robot_height],
                [init_state[0] - self.robot_width, init_state[1] + self.robot_height],
            ]
        )

        # init the robot
        self.robot = polytopic_robot.Polytopic_robot(0, self.robot_vertexes)
        # TODO consider orientation
        self.robot_init_state = self.robot.init_state
        self.robot_cur_state = np.copy(self.robot_init_state)
        self.robot_target_state = np.array(robot_params["target_state"])
        self.destination_margin = robot_params["destination_margin"]

        # initialize the circular-shaped obstacle
        self.cir_obs_states_list = None
        if cir_obs_params is not None:
            self.cir_obs_num = len(cir_obs_params["obs_states"])
            self.cir_obs_list = [None for i in range(self.cir_obs_num)]
            for i in range(self.cir_obs_num):
                self.cir_obs_list[i] = obs.Circle_Obs(
                    index=i,
                    radius=cir_obs_params["obs_radiuses"][i],
                    center=cir_obs_params["obs_states"][i],
                    vel=cir_obs_params["obs_vels"][i],
                    mode=cir_obs_params["modes"][i],
                )

            # get cir_obstacles' center position and velocity as well as radius
            self.cir_obs_init_states_list = [
                self.cir_obs_list[i].get_current_state()
                for i in range(self.cir_obs_num)
            ]
            self.cir_obs_states_list = np.copy(self.cir_obs_init_states_list)

        # initialize the other shaped obstacle state, vertexes
        self.obs_states_list = None
        if obs_params is not None:
            self.obs_num = len(obs_params["obs_vertexes"])
            self.obs_list = [None for i in range(self.obs_num)]
            for i in range(self.obs_num):
                self.obs_list[i] = obs.Polytopic_Obs(
                    index=i,
                    vertex=obs_params["obs_vertexes"][i],
                    vel=obs_params["obs_vels"][i],
                    mode=obs_params["modes"][i],
                )
            # get obstacles' center position and velocity
            self.obs_init_states_list = [
                self.obs_list[i].get_current_state() for i in range(self.obs_num)
            ]
            self.obs_states_list = np.copy(self.obs_init_states_list)
            self.obs_init_vertexes_list = [
                self.obs_list[i].init_vertexes for i in range(self.obs_num)
            ]

        # controller
        self.T = controller_params["Tmax"]
        self.step_time = controller_params["step_time"]
        self.time_steps = int(np.ceil(self.T / self.step_time))
        self.terminal_time = self.time_steps

        # storage
        self.xt = np.zeros((3, self.time_steps + 1))
        self.ut = np.zeros((2, self.time_steps))
        if obs_params is not None:
            self.obstacle_state_t = np.zeros((self.obs_num, 4, self.time_steps + 1))
            self.obs_cbf_t = np.zeros((self.obs_num, 1, self.time_steps))
            # for plot
            self.obs = [None for i in range(self.obs_num)]
        if cir_obs_params is not None:
            self.cir_obstacle_state_t = np.zeros(
                (self.cir_obs_num, 5, self.time_steps + 1)
            )
            self.cir_obs_cbf_t = np.zeros((self.cir_obs_num, 1, self.time_steps))
            # for plot
            self.cir_obs = [None for i in range(self.cir_obs_num)]

        self.clf1t = np.zeros((1, self.time_steps))
        self.clf2t = np.zeros((1, self.time_steps))
        self.slack1t = np.zeros((1, self.time_steps))
        self.slack2t = np.zeros((1, self.time_steps))

        # plot
        self.fig, self.ax = plt.subplots()

        # start and end state of robot
        self.start_body = None
        self.start_arrow = None
        self.end_body = None

        self.robot_body = None
        self.robot_arrow = None
        self.show_obs = True

    def navigation_destination(self):
        """navigate the robot to its destination"""
        t = 0
        process_time = []
        # approach the destination or exceed the maximum time
        while (
            np.linalg.norm(self.robot_cur_state[0:2] - self.robot_target_state[0:2])
            >= self.destination_margin
            and t - self.time_steps < 0.0
        ):
            start_time = time.time()
            u, clf1, clf2, feas = self.cbf_qp.clf_qp(self.robot_cur_state)
            process_time.append(time.time() - start_time)
            if not feas:
                print("This problem is infeasible, we can not get a feasible solution!")
                break
            else:
                pass

            self.xt[:, t] = np.copy(self.robot_cur_state)
            self.ut[:, t] = u
            self.clf1t[:, t] = clf1
            self.clf2t[:, t] = clf2
            # update the state of robot
            self.robot_cur_state = self.cbf_qp.robot.next_state(
                self.robot_cur_state, u, self.step_time
            )

            t = t + 1
        self.terminal_time = t
        # storage the last state of robot
        self.xt[:, t] = np.copy(self.robot_cur_state)

        self.show_obs = False
        print("Total time: ", self.terminal_time)
        print("Finish the solve of qp with clf!")
        print("Average_time:", sum(process_time) / len(process_time))

    def collision_avoidance(self):
        """solve the collision avoidance between robot and obstacles based on sdf-cbf"""
        t = 0
        process_time = []
        # approach the destination or exceed the maximum time
        while (
            np.linalg.norm(self.robot_cur_state[0:2] - self.robot_target_state[0:2])
            >= self.destination_margin
            and t - self.time_steps < 0.0
        ):
            # assign nominal controls （usually for no clf）
            u_ref = np.array([0.0, 0.0])
            add_clf = True

            # get the current optimal controls
            obs_vertexes_list = None
            if self.obs_states_list is not None:
                obs_vertexes_list = [
                    self.obs_list[i].vertexes for i in range(self.obs_num)
                ]

            start_time = time.time()
            (
                u,
                clf1,
                clf2,
                slack1,
                slack2,
                feas,
                cbf_list,
                cir_cbf_list,
            ) = self.cbf_qp.cbf_clf_qp(
                self.robot_cur_state,
                self.obs_states_list,
                obs_vertexes_list,
                self.cir_obs_states_list,
                add_clf=add_clf,
                u_ref=u_ref,
            )
            process_time.append(time.time() - start_time)
            if not feas:
                print("This problem is infeasible, we can not get a feasible solution!")
                break
            else:
                pass

            self.ut[:, t] = u
            self.clf1t[:, t] = clf1
            self.clf2t[:, t] = clf2
            self.slack1t[:, t] = slack1
            self.slack2t[:, t] = slack2

            # storage and update the state of robot and obstacle
            self.xt[:, t] = np.copy(self.robot_cur_state)
            self.robot_cur_state = self.cbf_qp.robot.next_state(
                self.robot_cur_state, u, self.step_time
            )
            if self.obs_states_list is not None:
                for i in range(self.obs_num):
                    self.obs_cbf_t[i][:, t] = cbf_list[i]
                    self.obstacle_state_t[i][:, t] = np.copy(self.obs_states_list[i])
                    self.obs_list[i].move_forward(self.step_time)
                self.obs_states_list = [
                    self.obs_list[i].get_current_state() for i in range(self.obs_num)
                ]

            if self.cir_obs_states_list is not None:
                for i in range(self.cir_obs_num):
                    self.cir_obs_cbf_t[i][:, t] = cir_cbf_list[i]
                    self.cir_obstacle_state_t[i][:, t] = np.copy(
                        self.cir_obs_states_list[i]
                    )
                    self.cir_obs_list[i].move_forward(self.step_time)
                self.cir_obs_states_list = [
                    self.cir_obs_list[i].get_current_state()
                    for i in range(self.cir_obs_num)
                ]
            t = t + 1
        self.terminal_time = t

        # storage the last state of robot and obstacles
        self.xt[:, t] = np.copy(self.robot_cur_state)
        if self.obs_states_list is not None:
            for i in range(self.obs_num):
                self.obstacle_state_t[i][:, t] = np.copy(self.obs_states_list[i])
        if self.cir_obs_states_list is not None:
            for i in range(self.cir_obs_num):
                self.cir_obstacle_state_t[i][:, t] = np.copy(
                    self.cir_obs_states_list[i]
                )

        print("Total time: ", self.terminal_time)
        print("Finish the solve of qp with sdf-cbf and clf!")
        print("Average_time:", sum(process_time) / len(process_time))

    def render(self):
        """Visualization"""
        self.fig.set_size_inches(7, 6.5)
        self.ax.set_aspect("equal")

        # set the text in Times New Roman
        config = {
            "font.family": "serif",
            "font.size": 12,
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
        }
        plt.rcParams.update(config)
        self.ax.set_xlim(-5, 20.0)
        self.ax.set_ylim(-5, 20.0)

        # set the label in Times New Roman and size
        label_font = {
            "family": "Times New Roman",
            "weight": "normal",
            "size": 16,
        }
        self.ax.set_xlabel("x (m)", label_font)
        self.ax.set_ylabel("y (m)", label_font)

        # set the tick in Times New Roman and size
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname("Times New Roman") for label in labels]

        self.animation_init()

        # robot and the arrow
        init_vertexes = self.robot.get_vertexes(self.robot_init_state)
        self.robot_body = mpatches.Polygon(
            init_vertexes, edgecolor="silver", facecolor=None
        )
        self.ax.add_patch(self.robot_body)

        self.robot_arrow = mpatches.Arrow(
            self.robot_init_state[0],
            self.robot_init_state[1],
            self.robot_width * np.cos(self.robot_init_state[2]),
            self.robot_width * np.sin(self.robot_init_state[2]),
            width=0.15,
            color="k",
        )
        self.ax.add_patch(self.robot_arrow)

        # show obstacles
        if self.show_obs:
            if self.obs_states_list is not None:
                for i in range(self.obs_num):
                    obs_vertexes = self.obs_list[i].get_current_vertexes(
                        self.obs_init_states_list[i]
                    )
                    self.obs[i] = mpatches.Polygon(obs_vertexes, color="k")
                    self.ax.add_patch(self.obs[i])
            if self.cir_obs_states_list is not None:
                for i in range(self.cir_obs_num):
                    self.cir_obs[i] = mpatches.Circle(
                        xy=self.cir_obs_init_states_list[i][0:2],
                        radius=self.cir_obs_init_states_list[i][4],
                        color="k",
                    )
                    self.ax.add_patch(self.cir_obs[i])

        self.ani = animation.FuncAnimation(
            self.fig,
            func=self.animation_loop,
            frames=self.terminal_time + 1,
            init_func=self.animation_init,
            interval=200,
            repeat=False,
        )
        plt.grid("--")
        # writergif = animation.PillowWriter(fps=30)
        # self.ani.save('pig.gif', writer=writergif)

        # writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        # self.ani.save('scatter.gif', writer=writer)
        plt.show()

    def animation_init(self):
        """init the robot start and end position"""
        # start body and arrow
        start_vertexes = self.robot.get_vertexes(self.robot_init_state)
        self.start_body = mpatches.Polygon(
            start_vertexes, edgecolor="silver", facecolor=None
        )
        self.ax.add_patch(self.start_body)
        self.start_body.set_zorder(0)

        self.start_arrow = mpatches.Arrow(
            self.robot_init_state[0],
            self.robot_init_state[1],
            self.robot_width * np.cos(self.robot_init_state[2]),
            self.robot_width * np.sin(self.robot_init_state[2]),
            width=0.15,
            color="k",
        )
        self.ax.add_patch(self.start_arrow)
        self.start_arrow.set_zorder(1)

        # target position
        self.end_body = mpatches.Circle(
            (self.robot_target_state[0], self.robot_target_state[1]),
            radius=0.5,
            color="silver",
        )
        self.ax.add_patch(self.end_body)
        self.end_body.set_zorder(0)

        return self.ax.patches + self.ax.texts + self.ax.artists

    def animation_loop(self, indx):
        """loop for update the position of robot and obstacles"""
        self.robot_body.remove()
        self.robot_arrow.remove()
        if self.show_obs:
            if self.obs_states_list is not None:
                for i in range(self.obs_num):
                    self.obs[i].remove()
            if self.cir_obs_states_list is not None:
                for i in range(self.cir_obs_num):
                    self.cir_obs[i].remove()

        # add robot and arrow
        cur_vertexes = self.robot.get_vertexes(self.xt[:, indx])
        self.robot_body = mpatches.Polygon(cur_vertexes, edgecolor="r", facecolor=None)
        self.ax.add_patch(self.robot_body)

        self.robot_arrow = mpatches.Arrow(
            self.xt[:, indx][0],
            self.xt[:, indx][1],
            self.robot_width * np.cos(self.xt[:, indx][2]),
            self.robot_width * np.sin(self.xt[:, indx][2]),
            width=0.15,
            color="k",
        )
        self.ax.add_patch(self.robot_arrow)

        # add obstacles
        if self.show_obs:
            if self.obs_states_list is not None:
                for i in range(self.obs_num):
                    obs_vertexes = self.obs_list[i].get_current_vertexes(
                        self.obstacle_state_t[i][:, indx]
                    )
                    self.obs[i] = mpatches.Polygon(obs_vertexes, color="k")
                    self.ax.add_patch(self.obs[i])

            if self.cir_obs_states_list is not None:
                for i in range(self.cir_obs_num):
                    self.cir_obs[i] = mpatches.Circle(
                        xy=self.cir_obstacle_state_t[i][:, indx][0:2],
                        radius=self.cir_obstacle_state_t[i][:, indx][4],
                        color="k",
                    )
                    self.ax.add_patch(self.cir_obs[i])

        # show past trajecotry of robot and obstacle
        if indx != 0:
            x_list = [self.xt[:, indx - 1][0], self.xt[:, indx][0]]
            y_list = [self.xt[:, indx - 1][1], self.xt[:, indx][1]]
            self.ax.plot(
                x_list,
                y_list,
                color="b",
            )

            # show past trajecotry of each dynamic obstacle
            if self.show_obs:
                if self.obs_states_list is not None:
                    for i in range(self.obs_num):
                        ox_list = [
                            self.obstacle_state_t[i][:, indx - 1][0],
                            self.obstacle_state_t[i][:, indx][0],
                        ]
                        oy_list = [
                            self.obstacle_state_t[i][:, indx - 1][1],
                            self.obstacle_state_t[i][:, indx][1],
                        ]
                        self.ax.plot(
                            ox_list,
                            oy_list,
                            linestyle="--",
                            color="k",
                        )
                if self.cir_obs_states_list is not None:
                    for i in range(self.cir_obs_num):
                        ox_list = [
                            self.cir_obstacle_state_t[i][:, indx - 1][0],
                            self.cir_obstacle_state_t[i][:, indx][0],
                        ]
                        oy_list = [
                            self.cir_obstacle_state_t[i][:, indx - 1][1],
                            self.cir_obstacle_state_t[i][:, indx][1],
                        ]
                        self.ax.plot(
                            ox_list,
                            oy_list,
                            linestyle="--",
                            color="k",
                        )

        plt.savefig('figure/{}.png'.format(indx), format='png', dpi=300)
        return self.ax.patches + self.ax.texts + self.ax.artists

    def show_control(self):
        """show controls"""
        # set the label in Times New Roman and size
        label_font = {
            "family": "Times New Roman",
            "weight": "normal",
            "size": 16,
        }

        t = np.arange(0, self.terminal_time / 10, self.step_time)

        plt.plot(
            t,
            self.ut[0][0 : self.terminal_time].reshape(
                self.terminal_time,
            ),
            linewidth=3,
            color="b",
            label="v",
        )
        plt.plot(t, self.u_max[0] * np.ones(t.shape[0]), "b--")
        plt.plot(t, self.u_min[0] * np.ones(t.shape[0]), "b--")

        plt.plot(
            t,
            self.ut[1][0 : self.terminal_time].reshape(
                self.terminal_time,
            ),
            linewidth=3,
            color="r",
            label="w",
        )
        plt.plot(t, self.u_max[1] * np.ones(t.shape[0]), "r--")
        plt.plot(t, self.u_min[1] * np.ones(t.shape[0]), "r--")

        plt.title("Control Variables", label_font)
        plt.xlabel("Time (s)", label_font)
        plt.ylabel("v (m/s) / w (rad/s)", label_font)

        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname("Times New Roman") for label in labels]

        legend_font = {"family": "Times New Roman", "weight": "normal", "size": 12}
        plt.legend(loc="upper right", prop=legend_font)
        plt.grid()
        plt.savefig("control.png", format="png", dpi=300)
        plt.show()
        plt.close(self.fig)

    def show_clf1(self):
        """show clf1"""
        # add an extra moment
        t = np.arange(0, self.terminal_time / 10, self.step_time)
        plt.plot(
            t,
            self.clf1t[:, 0 : self.terminal_time].reshape(self.terminal_time),
            linewidth=3,
            color="grey",
        )

        # set the label in Times New Roman and size
        label_font = {
            "family": "Times New Roman",
            "weight": "normal",
            "size": 16,
        }
        plt.title("CLF 1 for distance", label_font)
        plt.ylabel("clf ", label_font)
        plt.xlabel("Time (s)", label_font)

        # set the tick in Times New Roman and size
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname("Times New Roman") for label in labels]

        plt.grid()
        # plt.savefig('state_v.png', format='png', dpi=300)
        plt.show()
        plt.close(self.fig)

    def show_clf2(self):
        """show clf2"""
        # add an extra moment
        t = np.arange(0, self.terminal_time / 10, self.step_time)
        plt.plot(
            t,
            self.clf2t[:, 0 : self.terminal_time].reshape(self.terminal_time),
            linewidth=3,
            color="grey",
        )

        # set the label in Times New Roman and size
        label_font = {
            "family": "Times New Roman",
            "weight": "normal",
            "size": 16,
        }
        plt.title("CLF 2 for orientation", label_font)
        plt.ylabel("clf ", label_font)
        plt.xlabel("Time (s)", label_font)

        # set the tick in Times New Roman and size
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname("Times New Roman") for label in labels]

        plt.grid()
        # plt.savefig('state_v.png', format='png', dpi=300)
        plt.show()
        plt.close(self.fig)

    def show_slack(self):
        """show slack"""
        # add an extra moment
        t = np.arange(0, self.terminal_time / 10, self.step_time)
        plt.plot(
            t,
            self.slack1t[:, 0 : self.terminal_time].reshape(self.terminal_time),
            linewidth=3,
            color="grey",
        )
        plt.plot(
            t,
            self.slack2t[:, 0 : self.terminal_time].reshape(self.terminal_time),
            linewidth=3,
            color="blue",
        )

        # set the label in Times New Roman and size
        label_font = {
            "family": "Times New Roman",
            "weight": "normal",
            "size": 16,
        }
        plt.title("Slack", label_font)
        plt.ylabel("clf ", label_font)
        plt.xlabel("Time (s)", label_font)

        # set the tick in Times New Roman and size
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname("Times New Roman") for label in labels]

        plt.grid()
        # plt.savefig('state_v.png', format='png', dpi=300)
        plt.show()
        plt.close(self.fig)

    def show_cbf(self):
        """show cbf"""
        # add an extra moment
        t = np.arange(0, self.terminal_time / 10, self.step_time)
        plt.plot(
            t,
            self.obs_cbf_t[0][:, 0 : self.terminal_time].reshape(self.terminal_time),
            linewidth=3,
            color="grey",
        )

        # set the label in Times New Roman and size
        label_font = {
            "family": "Times New Roman",
            "weight": "normal",
            "size": 16,
        }
        plt.title("SDF with respect to Circle", label_font)
        plt.ylabel("cbf ", label_font)
        plt.xlabel("Time (s)", label_font)

        # set the tick in Times New Roman and size
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname("Times New Roman") for label in labels]

        plt.grid()
        # plt.savefig('state_v.png', format='png', dpi=300)
        plt.show()
        plt.close(self.fig)

    def show_cir_cbf(self):
        """show cbf"""
        # add an extra moment
        t = np.arange(0, self.terminal_time / 10, self.step_time)
        plt.plot(
            t,
            self.cir_obs_cbf_t[0][:, 0 : self.terminal_time].reshape(
                self.terminal_time
            ),
            linewidth=3,
            color="grey",
        )

        # set the label in Times New Roman and size
        label_font = {
            "family": "Times New Roman",
            "weight": "normal",
            "size": 16,
        }
        plt.title("SDF with respect to Polytope", label_font)
        plt.ylabel("cbf ", label_font)
        plt.xlabel("Time (s)", label_font)

        # set the tick in Times New Roman and size
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname("Times New Roman") for label in labels]

        plt.grid()
        # plt.savefig('state_v.png', format='png', dpi=300)
        plt.show()
        plt.close(self.fig)


if __name__ == "__main__":
    test_target = Collision_Avoidance()
    # test_target.navigation_destination()
    test_target.collision_avoidance()
    test_target.render()
    test_target.show_control()
    test_target.show_cir_cbf()
    # test_target.show_slack()
    # test_target.show_clf1()
    # test_target.show_clf2()
