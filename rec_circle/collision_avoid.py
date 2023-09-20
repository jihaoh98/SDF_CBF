import numpy as np
import time
import cbf_qp
import sdfmodel
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches


class Collision_Avoidance:
    def __init__(self) -> None:
        self.params = {
            'T': 20,
            'step_time': 0.1,
            'weight_input': [1.0, 1.0],
            'smooth_input': [1.0, 0.50],
            'weight_slack': 100,
            'clf_lambda': 0.75,
            'cbf_gamma': 1.0,
            'u_max': [2.0, 2.0],
            'u_min': [-2.0, -2.0],
            'initial_state': [2.0, 10.0],
            'target_state': [12.0, 10.0],
            'sensor_range': 6.0,
            'obstacle_state': [6.0, 9.0, 0.0, 0.0],
            'obstacle_radius': 0.85,
            'margin': 0.2,
            'destination_margin': 0.1,
            'width': 1.0,
            'height': 0.5,
            'e0': 1E-6
        }

        self.sdf_model = sdfmodel.Sdf_Model(self.params)
        self.cbf_qp = cbf_qp.Sdf_Cbf(self.sdf_model, self.params)

        # initial the robot state
        self.robot_init_state = np.array(self.params['initial_state'])
        self.robot_target_state = np.array(self.params['target_state'])
        self.robot_cur_state = np.copy(self.robot_init_state)
        self.robot_width = self.params['width']
        self.robot_height = self.params['height']
        self.destination_margin = self.params['destination_margin']

        # state of obstacle
        self.obstacle_init_state = np.array(self.params['obstacle_state'])
        self.obstacle_state = np.copy(self.obstacle_init_state)
        self.obstacle_radius = self.params['obstacle_radius']

        self.T = self.params['T']
        self.step_time = self.params['step_time']
        self.time_steps = int(np.ceil(self.T / self.step_time))
        self.terminal_time = self.time_steps

        # storage
        self.xt = np.zeros((2, self.time_steps + 1))
        self.ut = np.zeros((2, self.time_steps))
        self.obstacle_state_t = np.zeros((4, self.time_steps + 1))
        self.slackt = np.zeros((1, self.time_steps))
        self.clft = np.zeros((1, self.time_steps))
        self.cbft = np.zeros((1, self.time_steps))

        # plot
        self.fig, self.ax = plt.subplots()

        # start and end state of robot
        self.start_body = None
        self.end_body = None
        self.robot_body = None
        self.obs = None

    def collision_avoidance(self):
        """ solve the collision avoidance based on sdf-cbf """
        t = 0
        process_time = []
        # approach the destination or exceed the maximum time
        while np.linalg.norm(self.robot_cur_state - self.robot_target_state) >= self.destination_margin and t - self.time_steps < 0.0:
            # if t % 100 == 0:
            #     print(f't = {t}')

            start_time = time.time()
            u_ref = np.array([0.0, 0.0])
            u, clf, cbf, slack, feas = self.cbf_qp.cbf_clf_qp(self.robot_cur_state, self.obstacle_state, add_clf=True, u_ref=u_ref)
            process_time.append(time.time() - start_time)
            if not feas:
                print('This problem is infeasible, we can not get a feasible solution!')
                break
            else:
                pass

            self.xt[:, t] = np.copy(self.robot_cur_state)
            self.ut[:, t] = u
            self.clft[:, t] = clf
            self.cbft[:, t] = cbf
            self.slackt[:, t] = slack
                    
            # update the state of robot and obstacle
            self.robot_cur_state = self.sdf_model.next_state(self.robot_cur_state, u, self.step_time)
            self.obstacle_state_t[:, t] = np.copy(self.obstacle_state)
            self.obstacle_state[0:2] = self.obstacle_state[0:2] + self.obstacle_state[2:4] * self.step_time
            
            t = t + 1
        self.terminal_time = t 

        # storage the last time state of robot and obstacle
        self.xt[:, t] = np.copy(self.robot_cur_state)
        self.obstacle_state_t[:, t] = np.copy(self.obstacle_state)

        print('Total time: ', self.terminal_time)
        print('Finish the solve of qp with sdf-cbf and clf!')
        # print('Average_time:', sum(process_time) / len(process_time))

    def render(self):
        
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

        rectangle_position = self.robot_init_state - np.array([self.robot_width, self.robot_height])
        self.robot_body = mpatches.Rectangle((rectangle_position[0], rectangle_position[1]), self.robot_width * 2, self.robot_height * 2, edgecolor='silver', facecolor=None)
        self.ax.add_patch(self.robot_body)

        self.obs = mpatches.Circle(xy=(self.obstacle_init_state[0], self.obstacle_init_state[1]), radius=self.obstacle_radius, color='k')
        self.ax.add_patch(self.obs) 
        
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
        self.obs.remove()

        # add robot
        rectangle_position = self.xt[:, indx] - np.array([self.robot_width, self.robot_height])
        self.robot_body = mpatches.Rectangle(xy=(rectangle_position[0], rectangle_position[1]), 
                                             width=2 * self.robot_width, 
                                             height=2 * self.robot_height, 
                                             edgecolor='r', facecolor=None)
        self.ax.add_patch(self.robot_body)

        # add circle
        self.obs = mpatches.Circle(xy=(self.obstacle_state_t[:, indx][0], self.obstacle_state_t[:, indx][1]), 
                                   radius=self.obstacle_radius,
                                   color='k')
        self.ax.add_patch(self.obs)  

        # show past trajecotry of robot and obstacle
        if indx != 0:
            x_list = [self.xt[:, indx - 1][0], self.xt[:, indx][0]]
            y_list = [self.xt[:, indx - 1][1], self.xt[:, indx][1]]
            self.ax.plot(x_list, y_list, color='b',)

            ox_list = [self.obstacle_state_t[:, indx - 1][0], self.obstacle_state_t[:, indx][0]]
            oy_list = [self.obstacle_state_t[:, indx - 1][1], self.obstacle_state_t[:, indx][1]]    
            self.ax.plot(ox_list, oy_list, linestyle='--', color='k',)

        # plt.savefig('figure/{}.png'.format(indx), format='png', dpi=300)
        return self.ax.patches + self.ax.texts + self.ax.artists


if __name__ == '__main__':
    tese_target = Collision_Avoidance()
    tese_target.collision_avoidance()
    tese_target.render()







