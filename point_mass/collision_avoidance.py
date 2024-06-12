import numpy as np
from integral_sdf_qp import Integral_Sdf_Cbf_Clf
import time
import yaml
import obs
from render_show import Render_Animation
import matplotlib.pyplot as plt
import statistics


class Collision_Avoidance:
    def __init__(self, file_name) -> None:
        """ collision avoidance with obstacles """
        with open(file_name) as file:
            config = yaml.safe_load(file)
        
        robot_params = config['robot']
        controller_params = config['controller']
        self.cbf_qp = Integral_Sdf_Cbf_Clf(file_name)

        # init robot state
        self.robot_init_state = np.array(robot_params['initial_state'])
        self.robot_cur_state = np.copy(self.robot_init_state)
        self.robot_target_state = np.array(robot_params['target_state'])
        self.destination_margin = robot_params['destination_margin']

        # init obstacle, if no data, return None
        cir_obs_params = config.get('cir_obstacle_list')
        self.cir_obs_states_list = None
        if cir_obs_params is not None:
            self.cir_obs_num = len(cir_obs_params['obs_states'])
            self.cir_obs_list = [None for i in range(self.cir_obs_num)]
            for i in range(self.cir_obs_num):
                self.cir_obs_list[i] = obs.Circle_Obs(
                    index=i, 
                    radius=cir_obs_params['obs_radiuses'][i], 
                    center=cir_obs_params['obs_states'][i], 
                    vel=cir_obs_params['obs_vels'][i], 
                    mode=cir_obs_params['modes'][i],
                )
                            
            # get cir_obstacles' center position and velocity as well as radius
            self.cir_obs_init_states_list = [
                self.cir_obs_list[i].get_current_state() 
                for i in range(self.cir_obs_num)
            ]
            self.cir_obs_states_list = np.copy(self.cir_obs_init_states_list)

        # controller
        self.T = controller_params['Tmax']
        self.step_time = controller_params['step_time']
        self.time_steps = int(np.ceil(self.T / self.step_time))
        self.terminal_time = self.time_steps

        # storage
        self.xt = np.zeros((3, self.time_steps + 1))
        self.ut = np.zeros((3, self.time_steps))
        self.clft = np.zeros((1, self.time_steps))
        self.slackt = np.zeros((1, self.time_steps))

        self.cir_obstacle_state_t = None
        self.cir_obs_cbf_t = None
        if cir_obs_params is not None:
            self.cir_obstacle_state_t = np.zeros((self.cir_obs_num, 7, self.time_steps + 1))
            self.cir_obs_cbf_t = np.zeros((self.cir_obs_num, self.time_steps))

        # plot
        # self.ani = Render_Animation(
        #     robot_params, 
        #     cir_obs_params, 
        #     self.step_time,
        # )
        self.show_obs = True
    
    def navigation_destination(self, add_slack=False):
        """ navigate the robot to its destination """
        t = 0
        process_time = []
        # approach the destination or exceed the maximum time
        while (
            np.linalg.norm(self.robot_cur_state[0:3] - self.robot_target_state[0:3])
            >= self.destination_margin
            and t - self.time_steps < 0.0
        ):
            if t % 100 == 0:
                print(f't = {t}')
        
            start_time = time.time()
            optimal_result = self.cbf_qp.clf_qp(self.robot_cur_state, add_slack=add_slack)
            process_time.append(time.time() - start_time)

            if not optimal_result.feas:
                print('This problem is infeasible, we can not get a feasible solution!')
                break
            else:
                pass

            self.xt[:, t] = np.copy(self.robot_cur_state)
            self.ut[:, t] = optimal_result.u
            self.clft[0, t] = optimal_result.clf

            # update the state of robot
            self.robot_cur_state = self.cbf_qp.robot.next_state(self.robot_cur_state, optimal_result.u, self.step_time)
            t = t + 1

        self.terminal_time = t 
        # storage the last state of robot
        self.xt[:, t] = np.copy(self.robot_cur_state)
        self.show_obs = False

        print('Total time: ', self.terminal_time)
        if np.linalg.norm(self.robot_cur_state[0:3] - self.robot_target_state[0:3]) <= self.destination_margin:
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
            np.linalg.norm(self.robot_cur_state[0:3] - self.robot_target_state[0:3])
            >= self.destination_margin
            and t - self.time_steps < 0.0
        ):
            if t % 100 == 0:
                print(f't = {t}')

            start_time = time.time()
            optimal_result = self.cbf_qp.cbf_clf_qp(self.robot_cur_state, self.cir_obs_states_list, add_clf=add_clf)
            process_time.append(time.time() - start_time)

            if not optimal_result.feas:
                print('This problem is infeasible, we can not get a feasible solution!')
                break
            else:
                pass

            self.ut[:, t] = optimal_result.u
            self.clft[0, t] = optimal_result.clf
            self.slackt[0, t] = optimal_result.slack

            # storage and update the state of robot and obstacle
            self.xt[:, t] = np.copy(self.robot_cur_state)
            self.robot_cur_state = self.cbf_qp.robot.next_state(self.robot_cur_state, optimal_result.u, self.step_time)

            if self.cir_obs_states_list is not None:
                self.cir_obs_cbf_t[:, t] = optimal_result.cir_cbf_list
                for i in range(self.cir_obs_num):
                    self.cir_obstacle_state_t[i][:, t] = np.copy(self.cir_obs_states_list[i])
                    self.cir_obs_list[i].move_forward(self.step_time)
                self.cir_obs_states_list = [self.cir_obs_list[i].get_current_state() for i in range(self.cir_obs_num)]
            t = t + 1

        self.terminal_time = t 
        
        # storage the last state of robot and obstacles
        self.xt[:, t] = np.copy(self.robot_cur_state)
        if self.cir_obs_states_list is not None:
            for i in range(self.cir_obs_num):
                self.cir_obstacle_state_t[i][:, t] = np.copy(self.cir_obs_states_list[i])

        print('Total time: ', self.terminal_time)
        if np.linalg.norm(self.robot_cur_state[0:3] - self.robot_target_state[0:3]) <= self.destination_margin:
            print('Robot has arrived its destination!')
        else:
            print('Robot has not arrived its destination!')
        print('Finish the solve of QP with clf and cbf!')

        print('Maxinum_time:', max(process_time))
        print('Minimum_time:', min(process_time))
        print('Median_time:', statistics.median(process_time))
        print('Average_time:', statistics.mean(process_time))    

        for i in range(self.terminal_time + 1):
            print()
    
        
    def render(self):
        self.ani.render(self.xt, self.cir_obstacle_state_t, self.terminal_time, self.show_obs)
    
    def show_cbf(self, i):
        self.ani.show_cbf(i, self.cir_obs_cbf_t, self.terminal_time)

    def show_controls(self):
        self.ani.show_integral_controls(self.ut, self.terminal_time)

    def show_clf(self):
        self.ani.show_clf(self.clft[0], self.terminal_time)

    def show_slack(self):
        self.ani.show_slack(self.slackt[0], self.terminal_time)

if __name__ == '__main__':
    # file_name = 'dynamic_setting.yaml'
    file_name = 'static_setting.yaml'

    test_target = Collision_Avoidance(file_name)
    # test_target.navigation_destination()
    
    test_target.collision_avoidance()

    # test_target.render()
    # test_target.show_controls()
    # test_target.show_clf()
    # test_target.show_slack()
    # test_target.show_cbf(0)
    