import numpy as np
import time
import yaml
import statistics

from render_show import Render_Animation
from polytopic_obs import Polytopic_Obs
from polytopic_robot import Polytopic_Robot
from scaled_cbf_qp import Scaled_Cbf


class Collision_Avoidance:
    def __init__(self, file_name) -> None:
        """ collision avoidance with obstacles """
        with open(file_name) as file:
            config = yaml.safe_load(file)
        
        robot_params = config['robot']
        self.robot_model = robot_params['model']
        controller_params = config['controller']
        self.cbf_qp = Scaled_Cbf(file_name)

        # initialize the robot state, half width and height
        self.robot_model = robot_params['model']
        self.robot_width = robot_params['width']
        self.robot_height = robot_params['height']
        init_state = np.array(robot_params['initial_state'])
        robot_vertexes = np.array([
            [init_state[0] - self.robot_width, init_state[1] - self.robot_height],
            [init_state[0] + self.robot_width, init_state[1] - self.robot_height],
            [init_state[0] + self.robot_width, init_state[1] + self.robot_height],
            [init_state[0] - self.robot_width, init_state[1] + self.robot_height]
        ])
        self.robot = Polytopic_Robot(0, robot_vertexes)
        self.robot_target_state = np.array(robot_params['target_state'])
        self.destination_margin = robot_params['destination_margin']

        obs_params = config.get('obstacle_list')
        self.obs_num = len(obs_params['obs_vertexes'])
        self.obs_list = [None for i in range(self.obs_num)]
        for i in range(self.obs_num):
            self.obs_list[i] = Polytopic_Obs(
                indx=i, 
                vertex=obs_params['obs_vertexes'][i], 
                vel=obs_params['obs_vels'][i], 
                mode=obs_params['modes'][i],
            )

        # controller
        self.T = controller_params['Tmax']
        self.step_time = controller_params['step_time']
        self.time_steps = int(np.ceil(self.T / self.step_time))
        self.terminal_time = self.time_steps

        # storage
        self.xt = np.zeros((3, self.time_steps + 1))
        self.ut = np.zeros((2, self.time_steps))

        self.obstacle_state_t = np.zeros((self.obs_num, 4, self.time_steps + 1))
        self.obs_cbf_t = np.zeros((self.obs_num, self.time_steps))

        self.clft = np.zeros((2, self.time_steps))
        self.slackt = np.zeros((2, self.time_steps))

        # plot
        self.ani = Render_Animation(
            self.robot, 
            robot_params, 
            self.obs_list, 
            self.step_time,
        )
        self.show_obs = True
    
    def navigation_destination(self):
        """ navigate the robot to its destination """
        t = 0
        process_time = []

        # approach the destination or exceed the maximum time
        while (
            np.linalg.norm(self.robot.cur_state[0:2] - self.robot_target_state[0:2]) >= self.destination_margin
            and t - self.time_steps < 0.0
        ):
            if t % 100 == 0:
                print(f't = {t}')
        
            start_time = time.time()
            if self.robot_model == 'integral':
                u, clf, feas = self.cbf_qp.clf_qp(self.robot.cur_state)
            elif self.robot_model == 'unicycle':
                u, clf1, clf2, feas = self.cbf_qp.clf_qp(self.robot.cur_state)
            process_time.append(time.time() - start_time)

            if not feas:
                print('This problem is infeasible, we can not get a feasible solution!')
                break
            else:
                pass

            self.xt[:, t] = np.copy(self.robot.cur_state)
            self.ut[:, t] = u

            if self.robot_model == 'integral':
                self.clft[0, t] = clf
            elif self.robot_model == 'unicycle':
                self.clft[:, t] = np.array([clf1, clf2])

            # update the state of robot
            new_state = self.cbf_qp.robot.next_state(self.robot.cur_state, u, self.step_time)
            self.robot.update_state(new_state)
            t = t + 1

        self.terminal_time = t 
        # storage the last state of robot
        self.xt[:, t] = np.copy(self.robot.cur_state)
        self.show_obs = False

        print('Total time: ', self.terminal_time)
        if np.linalg.norm(self.robot.cur_state[0:2] - self.robot_target_state[0:2]) <= self.destination_margin:
            print('Robot has arrived its destination!')
        else:
            print('Robot has not arrived its destination!')
        print('Finish the solve of QP with clf!')

        print('Maxinum_time:', max(process_time))
        print('Minimum_time:', min(process_time))
        print('Median_time:', statistics.median(process_time))
        print('Average_time:', statistics.mean(process_time))

    def collision_avoidance(self, add_clf=True):
        """ solve the collision avoidance between robot and obstacles based on sdf-cbf """
        t = 0
        process_time = []

        # approach the destination or exceed the maximum time
        while (
            np.linalg.norm(self.robot.cur_state[0:2] - self.robot_target_state[0:2]) >= self.destination_margin
            and t - self.time_steps < 0.0
        ):
            if t % 100 == 0:
                print(f't = {t}')

            start_time = time.time()
            if self.robot_model == 'integral':
                u, cbf_list, clf, slack, feas = self.cbf_qp.cbf_clf_qp(
                    self.robot, self.obs_list, add_clf=add_clf
                )  
            elif self.robot_model == 'unicycle':
                u, cbf_list, clf1, clf2, slack1, slack2, feas = self.cbf_qp.cbf_clf_qp(
                    self.robot, self.obs_list, add_clf=add_clf
                ) 
            process_time.append(time.time() - start_time)

            if not feas:
                print('This problem is infeasible, we can not get a feasible solution!')
                break
            else:
                pass
            
            self.xt[:, t] = np.copy(self.robot.cur_state)
            self.ut[:, t] = u

            if self.robot_model == 'integral':
                self.clft[0, t] = clf
                self.slackt[0, t] = slack
            elif self.robot_model == 'unicycle':
                self.clft[:, t] = np.array([clf1, clf2])
                self.slackt[:, t] = np.array([slack1, slack2])

            # update the state of robot and obstacle
            new_state = self.cbf_qp.robot.next_state(self.robot.cur_state, u, self.step_time)
            self.robot.update_state(new_state)

            self.obs_cbf_t[:, t] = cbf_list
            for i in range(self.obs_num):
                self.obstacle_state_t[i][:, t] = np.copy(self.obs_list[i].get_current_state())
                self.obs_list[i].move_forward(self.step_time)
            t = t + 1

        self.terminal_time = t 
        # storage the last state of robot and obstacles
        self.xt[:, t] = np.copy(self.robot.cur_state)
        for i in range(self.obs_num):
            self.obstacle_state_t[i][:, t] = np.copy(self.obs_list[i].get_current_state())

        print('Total time: ', self.terminal_time)
        if np.linalg.norm(self.robot.cur_state[0:2] - self.robot_target_state[0:2]) <= self.destination_margin:
            print('Robot has arrived its destination!')
        else:
            print('Robot has not arrived its destination!')
        print('Finish the solve of QP with clf!')

        print('Maxinum_time:', max(process_time))
        print('Minimum_time:', min(process_time))
        print('Median_time:', statistics.median(process_time))
        print('Average_time:', statistics.mean(process_time))    
        
    def render(self):
        self.ani.render(self.xt, self.obstacle_state_t, self.terminal_time, self.show_obs)
    
    def show_cbf(self, i):
        # self.ani.show_both_cbf(i, self.obs_cbf_t, self.cir_obs_cbf_t, self.terminal_time)
        # self.ani.show_cbf(i, self.obs_cbf_t, self.terminal_time)
        self.ani.show_cbf(i, self.cir_obs_cbf_t, self.terminal_time)

    def show_controls(self):
        self.ani.show_integral_controls(self.ut, self.terminal_time)

    def show_clf(self):
        self.ani.show_clf('distance', self.clft[0], self.terminal_time)
        if self.robot_model == 'unicycle':
            self.ani.show_clf('theta', self.clft[1], self.terminal_time)

    def show_slack(self):
        self.ani.show_slack('distance', self.slackt[0], self.terminal_time)
        if self.robot_model == 'unicycle':
            self.ani.show_slack('theta', self.slackt[1], self.terminal_time)

if __name__ == '__main__':
    # file_name = 'integral_setting.yaml'
    file_name = 'unicycle_setting.yaml'

    test_target = Collision_Avoidance(file_name)
    # test_target.navigation_destination()
    test_target.collision_avoidance()
    test_target.render()

    