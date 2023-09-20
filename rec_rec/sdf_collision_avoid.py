import numpy as np
import sdf_cbf_qp
import obs
import time
import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches


class Collision_Avoidance:
    def __init__(self) -> None:

        file_name = 'settings.yaml'
        with open(file_name) as file:
            config = yaml.safe_load(file)
        
        robot_params = config['robot']
        obs_params = config['obstacle_list']
        controller_params = config['controller']
        self.cbf_qp = sdf_cbf_qp.Sdf_Cbf()

        # initialize the robot state, half width and height
        self.robot_init_state = np.array(robot_params['initial_state'])
        self.robot_target_state = np.array(robot_params['target_state'])
        self.robot_cur_state = np.copy(self.robot_init_state)
        self.robot_width = robot_params['width']
        self.robot_height = robot_params['height']
        self.destination_margin = robot_params['destination_margin']

        # initialize the obstacle state, vertexes
        self.obs_num = len(obs_params['obs_vertexes'])
        self.obs_list = [None for i in range(self.obs_num)]
        for i in range(self.obs_num):
            self.obs_list[i] = obs.Polytopic_Obs(index=i, vertex=obs_params['obs_vertexes'][i], 
                                                 vel=obs_params['obs_vels'][i], obs_model=obs_params['obs_models'][i])
        # get obstacles' center position and velocity
        self.obs_init_states_list = [self.obs_list[i].get_current_state() for i in range(self.obs_num)]
        self.obs_states_list = np.copy(self.obs_init_states_list)
        self.obs_init_vertexes_list = [self.obs_list[i].init_vertexes for i in range(self.obs_num)]

        # controller
        self.T = controller_params['T']
        self.step_time = controller_params['step_time']
        self.time_steps = int(np.ceil(self.T / self.step_time))
        self.terminal_time = self.time_steps

        # storage
        self.xt = np.zeros((2, self.time_steps + 1))
        self.ut = np.zeros((2, self.time_steps))
        self.obstacle_state_t = np.zeros((self.obs_num, 4, self.time_steps + 1))
        self.slackt = np.zeros((1, self.time_steps))
        self.clft = np.zeros((1, self.time_steps))

        # plot
        self.fig, self.ax = plt.subplots()

        # start and end state of robot
        self.start_body = None
        self.end_body = None
        self.robot_body = None
        self.obs = [None for i in range(self.obs_num)]

    def collision_avoidance(self):
        """ solve the collision avoidance between rec to rec based on sdf-cbf """
        t = 0
        process_time = []
        # approach the destination or exceed the maximum time
        while np.linalg.norm(self.robot_cur_state - self.robot_target_state) >= self.destination_margin and t - self.time_steps < 0.0:

            start_time = time.time()

            u_ref = np.array([0.0, 0.0])
            # get each obs's state and vertexes
            obs_vertexes_list = [self.obs_list[i].vertexes for i in range(self.obs_num)]
            u, clf, slack, feas = self.cbf_qp.cbf_clf_qp(self.robot_cur_state, obs_vertexes_list, self.obs_states_list, add_clf=True, u_ref=u_ref)
            process_time.append(time.time() - start_time)
            if not feas:
                print('This problem is infeasible, we can not get a feasible solution!')
                break
            else:
                pass

            self.xt[:, t] = np.copy(self.robot_cur_state)
            self.ut[:, t] = u
            self.clft[:, t] = clf
            self.slackt[:, t] = slack
                    
            # update the state of robot and obstacle
            self.robot_cur_state = self.cbf_qp.robot.next_state(self.robot_cur_state, u, self.step_time)
            # storage the states of obs and update its state
            for i in range(self.obs_num):
                self.obstacle_state_t[i][:, t] = np.copy(self.obs_states_list[i])
                self.obs_list[i].move_forward(self.step_time)

            self.obs_states_list = [self.obs_list[i].get_current_state() for i in range(self.obs_num)]
            t = t + 1
        self.terminal_time = t 

        # storage the last state of robot and obstacle
        self.xt[:, t] = np.copy(self.robot_cur_state)
        for i in range(self.obs_num):
            self.obstacle_state_t[i][:, t] = np.copy(self.obs_states_list[i])

        print('Total time: ', self.terminal_time)
        print('Finish the solve of qp with sdf-cbf and clf!')
        print('Average_time:', sum(process_time) / len(process_time))

    def render(self):
        """ Visualization """
        self.fig.set_size_inches(7, 6.5)
        self.ax.set_aspect('equal')

        # set the text in Times New Roman
        config = {
            "font.family": 'serif',
            "font.size": 12,
            "font.serif": ['Times New Roman'],
            "mathtext.fontset": 'stix',
        }
        plt.rcParams.update(config)
        self.ax.set_xlim(-0.8, 20.0)
        self.ax.set_ylim(-0.8, 20.0)

        # set the label in Times New Roman and size
        label_font = {'family': 'Times New Roman',
                      'weight': 'normal',
                      'size': 16,
                      }
        self.ax.set_xlabel('x (m)', label_font)
        self.ax.set_ylabel("y (m)", label_font)

        # set the tick in Times New Roman and size
        self.ax.tick_params(labelsize=16)
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        self.animation_init()

        # robot
        rectangle_position = self.robot_init_state - np.array([self.robot_width, self.robot_height])
        self.robot_body = mpatches.Rectangle((rectangle_position[0], rectangle_position[1]), self.robot_width * 2, self.robot_height * 2, edgecolor='silver', facecolor=None)
        self.ax.add_patch(self.robot_body)

        # obstacle
        for i in range(self.obs_num):
            obs_vertexes = self.obs_list[i].get_current_vertexes(self.obs_init_states_list[i])
            self.obs[i] = mpatches.Polygon(obs_vertexes, color='k')
            self.ax.add_patch(self.obs[i]) 

        self.ani = animation.FuncAnimation(self.fig,
                                           func=self.animation_loop,
                                           frames=self.terminal_time + 1,
                                           init_func=self.animation_init,
                                           interval=200,
                                           repeat=False)
        plt.grid('--')
        # writergif = animation.PillowWriter(fps=30) 
        # self.ani.save('pig.gif', writer=writergif)

        # writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        # ani.save('scatter.gif', writer=writer)
        plt.show()

    def animation_init(self):
        """ init the robot start and end position """
        # start position
        start_rec = self.robot_init_state - np.array([self.robot_width, self.robot_height])
        self.start_body = mpatches.Rectangle((start_rec[0], start_rec[1]), self.robot_width * 2, self.robot_height * 2, edgecolor='silver', facecolor=None)
        self.ax.add_patch(self.start_body)
        self.start_body.set_zorder(0)

        # target position
        end_rec = self.robot_target_state - np.array([self.robot_width, self.robot_height])
        self.end_body = mpatches.Rectangle((end_rec[0], end_rec[1]), self.robot_width * 2, self.robot_height * 2, edgecolor='silver', facecolor=None)
        self.ax.add_patch(self.end_body)
        self.end_body.set_zorder(0)
    
        return self.ax.patches + self.ax.texts + self.ax.artists

    def animation_loop(self, indx):
        """ loop for update the position of robot and obstacle """
        self.robot_body.remove()
        for i in range(self.obs_num):
            self.obs[i].remove()

        # add robot
        rectangle_position = self.xt[:, indx] - np.array([self.robot_width, self.robot_height])
        self.robot_body = mpatches.Rectangle(xy=(rectangle_position[0], rectangle_position[1]), 
                                             width=2 * self.robot_width, 
                                             height=2 * self.robot_height, 
                                             edgecolor='r', facecolor=None)
        self.ax.add_patch(self.robot_body)

        # add obstacle
        for i in range(self.obs_num):
            obs_vertexes = self.obs_list[i].get_current_vertexes(self.obstacle_state_t[i][:, indx])
            self.obs[i] = mpatches.Polygon(obs_vertexes, color='k')
            self.ax.add_patch(self.obs[i])

        # show past trajecotry of robot and obstacle
        if indx != 0:
            x_list = [self.xt[:, indx - 1][0], self.xt[:, indx][0]]
            y_list = [self.xt[:, indx - 1][1], self.xt[:, indx][1]]
            self.ax.plot(x_list, y_list, color='b',)

            # show past trajecotry of each obstacle
            for i in range(self.obs_num):
                ox_list = [self.obstacle_state_t[i][:, indx - 1][0], self.obstacle_state_t[i][:, indx][0]]
                oy_list = [self.obstacle_state_t[i][:, indx - 1][1], self.obstacle_state_t[i][:, indx][1]]  
                self.ax.plot(ox_list, oy_list, linestyle='--', color='k',)

        # plt.savefig('figure/{}.png'.format(indx), format='png', dpi=300)
        return self.ax.patches + self.ax.texts + self.ax.artists


if __name__ == '__main__':
    test_target = Collision_Avoidance()
    test_target.collision_avoidance()
    test_target.render()