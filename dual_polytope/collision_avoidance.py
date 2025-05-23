import time
import yaml
import numpy as np
import statistics
import pypoman

from render_show import Render_Animation
from polytopic_obs import Polytopic_Obs
from polytopic_robot import Polytopic_Robot
from scaled_cbf_qp import Scaled_Cbf
import matplotlib.pyplot as plt

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
        self.robot = Polytopic_Robot(0, np.array(robot_params['initial_state']))
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
        # self.ani = Render_Animation(
        #     self.robot, 
        #     robot_params, 
        #     self.obs_list, 
        #     self.step_time,
        # )
        # self.show_obs = True
    
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
        self.robot.A0 = np.vstack((np.eye(2), -np.eye(2)))
        self.robot.b0= np.array([0.5, 0.1, 0.5, 0.1]).reshape(4, 1)
        self.robot.G0 = np.vstack((np.eye(2), -np.eye(2)))
        self.robot.g0 = np.array([2.0, 2.0, -1.0, -1.0]).reshape(4, 1)
        self.robot.cur_state = np.copy(self.robot.init_state)
        fig, ax = plt.subplots(figsize=(10, 10))
        robot_vertices = np.array(pypoman.compute_polygon_hull(self.robot.A0, self.robot.b0.flatten()))
        robot_vertices_plot = np.vstack((robot_vertices, robot_vertices[0]))
        ax.plot(robot_vertices_plot[:, 0], robot_vertices_plot[:, 1], 'r-')
        ax.plot(self.robot_target_state[0], self.robot_target_state[1], 'ro')
        obs_vertices = np.array(pypoman.compute_polygon_hull(self.obs_list[0].A, self.obs_list[0].b.flatten()))
        obs_vertices_plot = np.vstack((obs_vertices, obs_vertices[0]))
        ax.plot(obs_vertices_plot[:, 0], obs_vertices_plot[:, 1], 'b-')

        plt.axis('equal')
        plt.grid('on')

        while (
            np.linalg.norm(self.robot.cur_state[0:2] - self.robot_target_state[0:2]) >= self.destination_margin
            and t - self.time_steps < 0.0
        ):
            if t % 100 == 0:
                print(f't = {t}')

            start_time = time.time()
            if self.robot_model == 'integral':
                u, cbf_list, clf, slack, feas = self.cbf_qp.cbfself._clf_qp(
                    self.robot, self.obs_list, add_clf=add_clf
                )  
            elif self.robot_model == 'unicycle':

                # self.robot.vertices = np.array(pypoman.compute_polygon_hull(self.robot.A, self.robot.b.flatten()))
                u, cbf_list, clf1, clf2, lam_i, lam_j, feas = self.cbf_qp.cbf_clf_qp(self.robot, add_clf=add_clf)

            process_time.append(time.time() - start_time)

            # show obstacle and robot 
            plt.pause(0.5)

            if t == 55:
                print('stop here')

            if not feas:
                print('This problem is infeasible, we can not get a feasible solution!')
                break
            else:
                pass
            
            self.xt[:, t] = np.copy(self.robot.cur_state)
            self.ut[:, t] = u


            # plot 
            s = self.robot.cur_state
            p = s[0:2]
            Rot = np.array([[np.cos(self.robot.cur_state[2]), -np.sin(self.robot.cur_state[2])],
                            [np.sin(self.robot.cur_state[2]), np.cos(self.robot.cur_state[2])]])
            A_cur = self.robot.A0 @ Rot.T 
            b_cur = self.robot.b0 + A_cur @ p.reshape(2, 1)
            rob_cur_vertices = np.array(pypoman.compute_polygon_hull(A_cur, b_cur.flatten()))
            robot_vertices_plot = np.vstack((rob_cur_vertices, rob_cur_vertices[0]))

            ax.plot(robot_vertices_plot[:, 0], robot_vertices_plot[:, 1], 'r-')


            if self.robot_model == 'integral':
                self.clft[0, t] = clf
                self.slackt[0, t] = slack
            elif self.robot_model == 'unicycle':
                self.clft[:, t] = np.array([clf1, clf2])
                # self.slackt[:, t] = np.array([slack1, slack2])

            # update the state of robot and obstacle
            new_state = self.cbf_qp.robot.next_state(self.robot.cur_state, u, self.step_time)
            # self.robot.update_state(new_state)
            self.robot.cur_state = new_state

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

    def storage_data(self, file_name):
        np.savez(
            file_name, xt=self.xt, ut=self.ut, 
            obs_cbf_t=self.obs_cbf_t,
            obs_list_t=self.obstacle_state_t, 
            ter=self.terminal_time
        )

    def load_data_unicycle_static(self):
        data = np.load('unicycle_static.npz')
        self.xt = data['xt']
        self.ut = data['ut']
        self.obstacle_state_t = data['obs_list_t']
        self.terminal_time = data['ter']

        self.ani.show_unicycle_model_controls(self.ut, self.terminal_time, name='controls_unicycle_static.png')
        # self.ani.show_unicycle_model(self.xt, self.obstacle_state_t, self.terminal_time, [32, 50])
        
    def load_data_unicycle_dynamic(self):
        data = np.load('unicycle_dynamic.npz')
        self.xt = data['xt']
        self.ut = data['ut']
        self.obstacle_state_t = data['obs_list_t']
        self.terminal_time = data['ter']
        # self.render()

        self.ani.show_unicycle_model_controls(self.ut, self.terminal_time)
        # self.ani.show_unicycle_model(self.xt, self.obstacle_state_t, self.terminal_time, [30, 42, 60, 70, 80])

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
    
    # test_target.load_data_unicycle_static()
    # test_target.load_data_unicycle_dynamic()
    test_target.render()
