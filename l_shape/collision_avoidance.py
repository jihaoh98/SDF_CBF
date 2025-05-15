import numpy as np
from integral_sdf_qp import Integral_Sdf_Cbf_Clf
from unicycle_sdf_qp import Unicycle_Sdf_Cbf_Clf
from obs import Circle_Obs, Polytopic_Obs
import l_shape_robot
import time
import yaml
import statistics
from render_show import Render_Animation
import casadi as ca
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt

class Collision_Avoidance:
    def __init__(self, file_name) -> None:
        """ collision avoidance with obstacles """
        with open(file_name) as file:
            config = yaml.safe_load(file)
        
        robot_params = config['robot']
        self.robot_model = robot_params['model']
        controller_params = config['controller']

        if self.robot_model == 'unicycle':
            self.cbf_qp = Unicycle_Sdf_Cbf_Clf(file_name)
        else:
            self.cbf_qp = Integral_Sdf_Cbf_Clf(file_name)

        # if no data, return None
        obs_params = config.get('obstacle_list')
        cir_obs_params = config.get('cir_obstacle_list')
        
        # initialize the robot 
        self.robot_vertexes = robot_params['vertexes']
        self.robot = l_shape_robot.L_shaped_robot(indx=0, init_state=[0.05, 1.5, np.pi/4], rects=self.robot_vertexes, mode='vertices')
        self.robot_init_state = self.robot.init_state
        self.robot_cur_state = np.copy(self.robot.init_state)
        self.robot_target_state = np.array(robot_params['target_state'])
        self.destination_margin = robot_params['destination_margin']

        # params for construct cbf
        # self.robot_params = np.array([self.robot.width, self.robot.height])
        # self.robot_two_center = self.robot.cur_center_body_frame.reshape(-1,)

        # initialize the circular-shaped obstacle
        self.cir_obs_states_list = None
        if cir_obs_params is not None:
            self.cir_obs_num = len(cir_obs_params['obs_states'])
            self.cir_obs_list = [None for i in range(self.cir_obs_num)]
            for i in range(self.cir_obs_num):
                self.cir_obs_list[i] = Circle_Obs(
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

        # initialize the other shaped obstacle state, vertexes
        self.obs_states_list = None
        self.obs_list = None
        if obs_params is not None:
            self.obs_num = len(obs_params['obs_vertexes'])
            self.obs_list = [None for i in range(self.obs_num)]
            for i in range(self.obs_num):
                self.obs_list[i] = Polytopic_Obs(
                    index=i, 
                    vertex=obs_params['obs_vertexes'][i], 
                    vel=obs_params['obs_vels'][i], 
                    mode=obs_params['modes'][i],
                )
            # get obstacles' center position and velocity
            self.obs_init_states_list = [
                self.obs_list[i].get_current_state() 
                for i in range(self.obs_num)
            ]
            self.obs_states_list = np.copy(self.obs_init_states_list)
            self.obs_init_vertexes_list = [
                self.obs_list[i].init_vertexes 
                for i in range(self.obs_num)
            ]

        # controller
        self.T = controller_params['Tmax']
        self.step_time = controller_params['step_time']
        self.time_steps = int(np.ceil(self.T / self.step_time))
        self.terminal_time = self.time_steps

        # storage
        self.xt = np.zeros((3, self.time_steps + 1))
        self.ut = np.zeros((2, self.time_steps))

        self.obstacle_state_t = None
        self.obs_cbf_t = None
        self.obs_cbf_t2 = None
        if obs_params is not None:
            self.obstacle_state_t = np.zeros((self.obs_num, 4, self.time_steps + 1))
            self.obs_cbf_t = np.zeros((self.obs_num, self.time_steps))
            self.obs_cbf_t2 = np.zeros((self.obs_num, self.time_steps))

        self.cir_obstacle_state_t = None
        self.cir_obs_cbf_t = None
        if cir_obs_params is not None:
            self.cir_obstacle_state_t = np.zeros((self.cir_obs_num, 5, self.time_steps + 1))
            self.cir_obs_cbf_t = np.zeros((self.cir_obs_num, self.time_steps))

        # consider unicycle model
        self.clft = np.zeros((2, self.time_steps))
        self.slackt = np.zeros((2, self.time_steps))

        # plot
        self.ani = Render_Animation(
            self.robot, 
            robot_params, 
            self.obs_list, 
            cir_obs_params, 
            self.step_time,
        )
        self.show_obs = True

    def convex_polygon_hrep(self, points):
        """
        Given a set of 2D points (vertices of a convex polygon or a point cloud),
        compute the H-representation (A, b) of the convex polygon such that Ax ≤ b 
        describes the polygon (each inequality corresponds to one edge).
        """
        # Convert input to a NumPy array (n_points x 2)
        pts = np.asarray(points, dtype=float)
        if pts.shape[1] != 2:
            raise ValueError("Input points must be 2-dimensional coordinates.")
        
        # 1. Compute the convex hull of the points
        hull = ConvexHull(pts)
        
        # The ConvexHull vertices are in counterclockwise order (for 2D):contentReference[oaicite:6]{index=6}.
        # We could use hull.vertices (indices of hull points) if needed for further processing.
        # Here, we'll use hull.equations to get the facet equations directly.
        
        # 2. Get the hyperplane equations for each facet (edge) of the hull.
        # hull.equations is an array of shape (n_facets, 3) for 2D: [a, b, c] for each line (a*x + b*y + c = 0).
        # For interior points of the hull, a*x + b*y + c ≤ 0 holds true:contentReference[oaicite:7]{index=7}.
        equations = hull.equations  # shape (n_edges, 3)
        
        # 3. Split each equation into normal vector (a, b) and offset c.
        A = equations[:, :2]   # all rows, first two columns -> coefficients [a, b] for x and y
        c = equations[:, 2]    # last column is c in a*x + b*y + c = 0
        
        # 4. Convert to inequality form: a*x + b*y ≤ -c
        # We move c to the right side: a*x + b*y ≤ -c.
        b = -c  # Now each inequality is [a, b] · [x, y] ≤ b_i (where b_i = -c).
        
        # At this point, each row of A and corresponding element of b represent 
        # an inequality defining the half-space that contains the convex polygon.
        # (The normal vectors in A point outward, and the interior of the polygon 
        # satisfies A*x ≤ b.)

        b = b.reshape(-1, 1)  # Reshape b to be a column vector (n_edges x 1)
        
        return A, b, hull

    def navigation_destination(self):
        """ navigate the robot to its destination """
        t = 0
        process_time = []
        # approach the destination or exceed the maximum time
        while (
            np.linalg.norm(self.robot_cur_state[0:2] - self.robot_target_state[0:2])
            >= self.destination_margin
            and t - self.time_steps < 0.0
        ):
            if t % 100 == 0:
                print(f't = {t}')
        
            start_time = time.time()
            if self.robot_model == 'integral':
                u, clf, feas = self.cbf_qp.clf_qp(self.robot_cur_state)
            elif self.robot_model == 'unicycle':
                u, clf1, clf2, feas = self.cbf_qp.clf_qp(self.robot_cur_state, add_slack=True)
            process_time.append(time.time() - start_time)

            if not feas:
                print('This problem is infeasible, we can not get a feasible solution!')
                break
            else:
                pass

            self.xt[:, t] = np.copy(self.robot_cur_state)
            self.ut[:, t] = u

            if self.robot_model == 'integral':
                self.clft[0, t] = clf
            elif self.robot_model == 'unicycle':
                self.clft[:, t] = np.array([clf1, clf2])

            # update the state of robot
            self.robot_cur_state = self.cbf_qp.robot.next_state(self.robot_cur_state, u, self.step_time)
            t = t + 1

        self.terminal_time = t 
        # storage the last state of robot
        self.xt[:, t] = np.copy(self.robot_cur_state)
        self.show_obs = False

        print('Total time: ', self.terminal_time)
        if np.linalg.norm(self.robot_cur_state[0:2] - self.robot_target_state[0:2]) <= self.destination_margin:
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
        # approach the destination or exceed the maximum timec

        cbf_1_list = []
        cbf_2_list = []
        while (
            np.linalg.norm(self.robot_cur_state[0:2] - self.robot_target_state[0:2])
            >= self.destination_margin
            and t - self.time_steps < 0.0
        ):
            if t % 100 == 0:
                print(f't = {t}')
            
            # get the current optimal control
            obs_vertexes_list = None
            if self.obs_states_list is not None:
                obs_vertexes_list = [self.obs_list[i].vertexes for i in range(self.obs_num)]

            start_time = time.time()

            robot_vertices_list = self.robot.get_vertices_at_absolute_state(self.robot_cur_state)
            # extract inequalities from the vertices
            mat_A, vec_a, _ = self.convex_polygon_hrep(robot_vertices_list[0])
            mat_B, vec_b, _ = self.convex_polygon_hrep(robot_vertices_list[1])
            mat_G, vec_g, _ = self.convex_polygon_hrep(obs_vertexes_list[0])

            # solve 2 x N QP, N depends on the number of obstacles
            opti = ca.Opti('conic');
            lam_A = opti.variable(1, 4)
            lam_AG = opti.variable(1, 4)
            obj = - 0.25 * lam_A @ mat_A @ mat_A.T @ lam_A.T - lam_A @ vec_a - lam_AG @ vec_g
            opti.minimize(-obj)  # max optimization
            opti.subject_to(lam_A @ mat_A + lam_AG @ mat_G == 0)
            opti.subject_to(lam_A >= 0)
            opti.subject_to(lam_AG >= 0)

            opti.solver('qpoases');  # the options should be alinged with the solver
            sol=opti.solve();
            lam_A_star = sol.value(lam_A).reshape(1, -1)
            lam_AG_star = sol.value(lam_AG).reshape(1, -1)
            lam_A_pos_idx = np.where(lam_A_star[0, :] < 1e-5)
            lam_AG_pos_idx = np.where(lam_AG_star[0, :] < 1e-5)
            dist_square = sol.value(obj)
            dist_AG = np.sqrt(dist_square)

            # # ======================= the second QP problem
            opti = ca.Opti('conic');
            lam_B = opti.variable(1, 4)
            lam_BG = opti.variable(1, 4)
            # the second qp w.r.t. rectangle B
            obj = - 0.25 * lam_B @ mat_B @ mat_B.T @ lam_B.T - lam_B @ vec_b - lam_BG @ vec_g
            opti.minimize(-obj)
            opti.subject_to(lam_B @ mat_B + lam_BG @ mat_G == 0)
            opti.subject_to(lam_B >= 0)
            opti.subject_to(lam_BG >= 0)

            opti.solver('qpoases')
            sol = opti.solve()
            lam_B_star = sol.value(lam_B).reshape(1, -1)
            lam_BG_star = sol.value(lam_BG).reshape(1, -1)
            lam_B_pos_idx = np.where(lam_B_star[0, :] < 1e-5)
            lam_BG_pos_idx = np.where(lam_BG_star[0, :] < 1e-5)
            dist_square = sol.value(obj)
            dist_BG = np.sqrt(dist_square)
            print('the AG distance is :', dist_AG)
            print('the BG distance is :', dist_BG)

            cbf_1_list.append(dist_AG)
            cbf_2_list.append(dist_BG)

            # exit()
            # ======================= the CLF-CBF-QP problem
            u, clf, slack, feas = self.cbf_qp.cbf_clf_qp(
                self.robot_cur_state,
                dist_AG, dist_BG,
                mat_A, vec_a, mat_B, vec_b, mat_G, vec_g,
                lam_A_star, lam_AG_star, lam_A_pos_idx, lam_AG_pos_idx,
                lam_B_star, lam_BG_star, lam_B_pos_idx, lam_BG_pos_idx,
                add_clf=add_clf)
            
            process_time.append(time.time() - start_time)

            if not feas:
                print('This problem is infeasible, we can not get a feasible solution!')
                break
            else:
                pass

            self.ut[:, t] = u
            self.clft[:, t] = np.array([clf])
            self.slackt[:, t] = np.array([slack])
            if self.obs_states_list is not None:
                self.obs_cbf_t[:, t] = dist_AG
                self.obs_cbf_t2[:, t] = dist_BG
                for i in range(self.obs_num):
                    self.obstacle_state_t[i][:, t] = np.copy(self.obs_states_list[i])
                    self.obs_list[i].move_forward(self.step_time)

                    # adjust
                    if self.robot_model == 'unicycle':
                        if self.obs_list[0].position[0] <= 0.5:
                            self.obs_list[0].vel[0] = 0.0

                self.obs_states_list = [self.obs_list[i].get_current_state() for i in range(self.obs_num)]

            # storage and update the state of robot and obstacle
            self.xt[:, t] = np.copy(self.robot_cur_state)
            self.robot_cur_state = self.cbf_qp.robot.next_state(self.robot_cur_state, u, self.step_time)
            t = t + 1

        self.terminal_time = t 
        # storage the last state of robot and obstacles
        self.xt[:, t] = np.copy(self.robot_cur_state)
        if self.obs_states_list is not None:
            for i in range(self.obs_num):
                self.obstacle_state_t[i][:, t] = np.copy(self.obs_states_list[i])

        print('Total time: ', self.terminal_time)
        if np.linalg.norm(self.robot_cur_state[0:2] - self.robot_target_state[0:2]) <= self.destination_margin:
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
            cir_obs_cbf_t=self.cir_obs_cbf_t,
            obs_list_t=self.obstacle_state_t, 
            cir_obs_list_t=self.cir_obstacle_state_t, 
            ter=self.terminal_time
        )
    
    def load_data_integral(self):
        data = np.load('integral_data.npz')
        self.xt = data['xt']
        self.ut = data['ut']
        self.obs_cbf_t = data['obs_cbf_t']
        self.cir_obs_cbf_t = data['cir_obs_cbf_t']
        self.obstacle_state_t = data['obs_list_t']
        self.cir_obstacle_state_t = data['cir_obs_list_t']
        self.terminal_time = data['ter']
        # self.show_cbf()
        self.show_controls()
        # self.ani.show_integral_model(self.xt, self.obstacle_state_t, self.cir_obstacle_state_t, self.terminal_time, [17, 34, 44])

    def load_data_unicycle(self):
        data = np.load('unicycle.npz')
        self.xt = data['xt']
        self.ut = data['ut']
        self.obs_cbf_t = data['obs_cbf_t']
        self.obstacle_state_t = data['obs_list_t']
        self.terminal_time = data['ter']
        # self.render()
        # self.ani.show_unicycle_cbf(self.obs_cbf_t, self.terminal_time)
        self.ani.show_unicycle_model_controls(self.ut, self.terminal_time)
        # self.ani.show_unicycle_model(self.xt, self.obstacle_state_t, self.terminal_time, [44, 102])

    def render(self):
        self.ani.render(self.xt, self.obstacle_state_t , self.terminal_time, self.show_obs)

    def show_cbf(self):
        self.ani.show_integral_cbf(self.obs_cbf_t, self.obs_cbf_t2, self.terminal_time)

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
    # file_name = 'integral_settings.yaml'
    file_name = 'unicycle_settings.yaml'
    test_target = Collision_Avoidance(file_name)
    # test_target.navigation_destination()
    test_target.collision_avoidance()
    # test_target.storage_data('integral_data.npz')
    # test_target.storage_data('unicycle.npz')

    # test_target.load_data_integral()
    # test_target.load_data_unicycle()
    test_target.render()
    # test_target.show_clf()
    # test_target.show_slack()
    # test_target.show_controls()
    test_target.show_cbf()

    # plot controls here
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(test_target.ut[0, :], label='u[0], v', color='blue')
    ax.plot(test_target.ut[1, :], label='u[1], w', color='red')
    plt.legend()
    plt.show()
