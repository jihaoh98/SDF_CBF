import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from polytopic_robot import Polytopic_Robot
from polytopic_obs import Polytopic_Obs


def derive_cbf_gradient(robot: Polytopic_Robot, obs: Polytopic_Obs):
    """ Derive the gradient of the CBF """
    R = np.array([
        [np.cos(robot.cur_state[2]), -np.sin(robot.cur_state[2])],
        [np.sin(robot.cur_state[2]), np.cos(robot.cur_state[2])]
    ])
    A_r = robot.A @ R.T
    robot_p = robot.cur_state[:2]
    br_initial = robot.b0

    A_obs = obs.A
    bo_initial = obs.b0
    b_obs = bo_initial + np.dot(A_obs, obs.cur_state[:2])

    x, beta = solve_scaled_lp(A_r, robot_p, br_initial, A_obs, b_obs)
    if x is None:
        return None, None, None
    
    # calculate the gradient of the CBF w.r.t the state
    ans_r = A_r @ (x - robot_p) - beta * br_initial
    index_r = np.where(np.abs(ans_r) < 1e-4)[0]

    ans_obs = A_obs @ x - b_obs
    index_obs = np.where(np.abs(ans_obs) < 1e-4)[0]

    # print(x, beta, index_r, index_obs)

    dBeta_dx, dBeta_dp_o = None, None
    if len(index_r) == 2:
        index_obs = index_obs[:1]
        dBeta_dx, dBeta_dp_o = gradient_case1(
            robot.A.copy(), br_initial, robot.cur_state, index_r, 
            obs.A.copy(), bo_initial, obs.cur_state, index_obs
        )
    elif len(index_r) == 1:
        dBeta_dx, dBeta_dp_o =  gradient_case2(
            robot.A.copy(), br_initial, robot.cur_state, index_r, 
            obs.A.copy(), bo_initial, obs.cur_state, index_obs
        )

    return beta, dBeta_dx, dBeta_dp_o
    
def solve_scaled_lp(A_r, robot_p, br_initial, A_obs, b_obs):
    """ Solve the scaled LP problem """
    opti = ca.Opti('conic')
    opts_setting = {
        'printLevel': 'low',  
        'error_on_fail': False,
        'expand': True,
        'print_time': 0
    }
    opti.solver('qpoases', opts_setting)

    x = opti.variable(2)
    beta = opti.variable()

    opti.minimize(beta)
    opti.subject_to(A_r @ (x - robot_p) <= beta * br_initial)
    opti.subject_to(A_obs @ x <= b_obs)
    opti.subject_to(beta >= 0)

    try:
        sol = opti.solve()
        return sol.value(x), sol.value(beta)
    except Exception as e:
        print('Failed to solve the LP problem:', e)
        return None, None
    
def gradient_case1(
        Ar_initial, br_initial, robot_cur_state, index_r, 
        A_obs, b_obs_initial, obs_cur_state, index_obs
    ):
    """ 2 ac of robot, 1 ac of obs """
    # Rotation matrix
    theta = robot_cur_state[2]
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    Ar = Ar_initial @ R.T

    A = np.vstack([Ar[index_r[0]], Ar[index_r[1]]])
    A_inverse = np.linalg.pinv(A)  
    B = np.array([-br_initial[index_r[0]], -br_initial[index_r[1]]]).reshape(-1, 1)
    C = np.array([A_obs[index_obs[0]]]).reshape(1, -1)
    E = np.array([
        Ar[index_r[0]] @ robot_cur_state[:2],
        Ar[index_r[1]] @ robot_cur_state[:2],     
    ]).reshape(-1, 1)
    F = A_obs[index_obs[0]] @ obs_cur_state[:2] + b_obs_initial[index_obs[0]]

    Lambda = C @ A_inverse @ E - F
    Theta = C @ A_inverse @ B
    # beta = Lambda / Theta
    # print('beta', beta[0, 0])

    # Compute dBeta / dp_r
    dE_dp_r = A.T
    dBeta_dp_r = dE_dp_r @ (C @ A_inverse).T / Theta

    # Compute dBeta / dtheta
    dR_T_dtheta = np.array([
        [-np.sin(theta), np.cos(theta)],
        [-np.cos(theta), -np.sin(theta)]
    ])
    dA_dtheta = np.vstack([Ar_initial[index_r[0]], Ar_initial[index_r[1]]]) @ dR_T_dtheta
    dA_inv_dtheta = -A_inverse @ dA_dtheta @ A_inverse  # d(A^-1)/dθ = - A^-1 (dA/dθ) A^-1

    # Compute dE / dθ
    dE_dtheta = dA_dtheta @ robot_cur_state[:2].reshape(-1, 1)

    # Compute dBeta / dθ
    dBeta_dtheta = (C @ dA_inv_dtheta @ E + C @ A_inverse @ dE_dtheta) / Theta - (Lambda / Theta**2) * (C @ dA_inv_dtheta @ B)
    
    # Compute dBeta / dx
    dBeta_dx = np.array([dBeta_dp_r[0, 0], dBeta_dp_r[1, 0], dBeta_dtheta[0, 0]]).reshape(-1, 1)
    
    # Compute dBeta / dp_o
    dBeta_dp_o = -A_obs[index_obs[0]].reshape(-1, 1) / Theta

    return dBeta_dx, dBeta_dp_o

def gradient_case2(
        Ar_initial, br_initial, robot_cur_state, index_r, 
        A_obs, b_obs, obs_cur_state, index_obs
    ):
    """  1 ac of robot, 2 ac of obs """
    # Rotation matrix
    theta = robot_cur_state[2]
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    Ar = Ar_initial @ R.T

    A = np.vstack([A_obs[index_obs[0]], A_obs[index_obs[1]]])  
    A_inverse = np.linalg.pinv(A)  
    C = np.array([Ar[index_r[0]]]).reshape(1, -1)
    D = -br_initial[index_r[0]]
    E = np.array([
        A_obs[index_obs[0]] @ obs_cur_state[:2] + b_obs[index_obs[0]],
        A_obs[index_obs[1]] @ obs_cur_state[:2] + b_obs[index_obs[1]]
    ]).reshape(-1, 1)
    F = Ar[index_r[0]] @ robot_cur_state[:2]

    # Compute beta
    # Lambda = C @ A_inverse @ E
    # beta = (F - Lambda) / D
    # print('beta', beta[0, 0])

    # Compute dBeta / dp_r
    dBeta_dp_r = R @ Ar_initial[index_r[0]].reshape(-1, 1) / D
    
    # Compute dBeta / dtheta
    dR_T_dtheta = np.array([
        [-np.sin(theta), np.cos(theta)],
        [-np.cos(theta), -np.sin(theta)]
    ])
    dBeta_dtheta = (1 / D) * (Ar_initial[index_r[0]] @ dR_T_dtheta @ (robot_cur_state[:2] - (A_inverse @ E).reshape(-1,)))

    # Compute dBeta / dp_o
    dE_dp_o = A.T
    dBeta_dp_o = (-dE_dp_o @ (C @ A_inverse).T) / D

    # Compute dBeta / dx
    dBeta_dx = np.array([dBeta_dp_r[0, 0], dBeta_dp_r[1, 0], dBeta_dtheta]).reshape(-1, 1)

    return dBeta_dx, dBeta_dp_o

def plot_polytope(vertices1, vertices2):
    fig, ax = plt.subplots()
    for i in range(len(vertices1)):
        p1 = vertices1[i]
        p2 = vertices1[(i + 1) % len(vertices1)]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r')
    
    for i in range(len(vertices2)):
        p1 = vertices2[i]
        p2 = vertices2[(i + 1) % len(vertices2)]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b')

    plt.axis('equal')
    plt.show()

def gradient_case1_test():
    robot_vertices = [[0.0, 1.0], [1.0, 0.5], [2.0, 1.0], [1.0, 1.5]]
    obs_vertices = [[3.0, 0.5], [5.0, 0.5], [5.0, 1.5], [3.0, 1.5]]

    robot = Polytopic_Robot(0, robot_vertices)
    obs = Polytopic_Obs(0, obs_vertices)

    # plot_polytope(robot.vertexes, obs.vertexes)
    beta, dBeta_dx, dBeta_dp_o = derive_cbf_gradient(robot, obs)
    print('beta:', beta)
    print('dBeta_dx:', dBeta_dx)
    print('dBeta_dp_o:', dBeta_dp_o)

def gradient_case2_test():
    robot_vertices = [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]
    obs_vertices = [[3.0, 1.5], [4.0, 1.0], [5.0, 1.5], [4.0, 2.0]]

    robot = Polytopic_Robot(0, robot_vertices)
    obs = Polytopic_Obs(0, obs_vertices)

    # plot_polytope(robot.vertexes, obs.vertexes)
    beta, dBeta_dx, dBeta_dp_o = derive_cbf_gradient(robot, obs)
    print('beta:', beta)
    print('dBeta_dx:', dBeta_dx)
    print('dBeta_dp_o:', dBeta_dp_o)

if __name__ == "__main__":
    # gradient_case1_test()
    gradient_case2_test()