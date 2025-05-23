import numpy as np
import pypoman
import matplotlib.pyplot as plt

def main():
    # A0 and b0 define a rectangle initially at position (0, 0) and with no rotation
    A0 = np.vstack((np.eye(2), -np.eye(2)))
    b0 = np.array([0.5, 0.1, 0.5, 0.1])

    vertices_0 =  pypoman.compute_polygon_hull(A0, b0)
    vertices_0_np = np.array(vertices_0)
    vertices_0_np_plot = np.vstack((vertices_0_np, vertices_0_np[0]))  # close the polygon

    fig, ax = plt.subplots()
    ax.plot(*zip(*vertices_0_np_plot), marker='o', color='blue', label='Original')
    plt.axis('equal')
    plt.grid('on')
    plt.show()

    # we can specify an initial state by using A0 @ R.T and b0 + A0 @ p_init
    s_init = np.array([1.5, 1.5, np.pi/4])
    p_init = s_init[:2]
    theta = s_init[2]

    R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    
    A_init = A0 @ R.T
    b_init = b0 + A_init @ p_init

    vertices_init = pypoman.compute_polygon_hull(A_init, b_init)
    vertices_init_np = np.array(vertices_init)
    vertices_init_np_plot = np.vstack((vertices_init_np, vertices_init_np[0]))  # close the polygon

    fig, ax = plt.subplots()
    ax.plot(*zip(*vertices_0_np_plot), marker='o', color='blue', label='Original')
    ax.plot(*zip(*vertices_init_np_plot), marker='*', color='red', label='Transformed')
    plt.scatter(p_init[0], p_init[1], color='green', label='Translation')
    plt.axis('equal')
    plt.grid('on')
    plt.legend()
    plt.show()

    # assume the sample_time is 0.1, the velocity is 0.1, and the angular velocity is 0.1
    dt = 0.3
    w = 0.5
    v = 3.0
    s_next = np.array([s_init[0] + dt * v * np.cos(s_init[2]),
                   s_init[1] + dt * v * np.sin(s_init[2]),
                   s_init[2] + dt * w])
    
    fig, ax = plt.subplots()
    ax.plot(*zip(*vertices_0_np_plot), marker='o', color='blue', label='Original')
    ax.plot(*zip(*vertices_init_np_plot), marker='*', color='red', label='Transformed')
    # plot the initial and next state
    ax.plot(s_init[0], s_init[1], 'go', label='Initial State')
    ax.plot(s_next[0], s_next[1], 'ro', label='Next State')
    plt.legend()
    plt.grid('on')
    plt.axis('equal')
    plt.show()

    # get vertices for the next state s_next or get A_next and b_next based on A_init and b_init
    theta_next = s_next[2]
    p_next = s_next[:2]

    R_next = np.array([[np.cos(theta_next), -np.sin(theta_next)],
                    [np.sin(theta_next),  np.cos(theta_next)]])

    A_next = A0 @ R_next.T
    b_next = b0 + A_next @ p_next

    vertices_next = pypoman.compute_polygon_hull(A_next, b_next)
    vertices_next_np = np.array(vertices_next)
    vertices_next_np_plot = np.vstack((vertices_next_np, vertices_next_np[0]))


    fig, ax = plt.subplots()
    ax.plot(*zip(*vertices_0_np_plot), marker='o', color='blue', label='Original')
    ax.plot(*zip(*vertices_init_np_plot), marker='*', color='red', label='s_init')
    ax.plot(*zip(*vertices_next_np_plot), marker='x', color='purple', label='s_next')

    ax.plot(s_init[0], s_init[1], 'go', label='Initial State')
    ax.plot(s_next[0], s_next[1], 'ro', label='Next State')

    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


    # 另一个方法
    # s_next = [x', y', theta']
    theta = s_next[2]
    p = s_next[:2]

    R = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]])

    # transform each vertex from the original shape
    vertices_next = [R @ v + p for v in vertices_0]
    vertices_next_np = np.array(vertices_next)
    vertices_next_np_plot = np.vstack((vertices_next_np, vertices_next_np[0]))  # close the polygon

    fig, ax = plt.subplots()
    ax.plot(*zip(*vertices_0_np_plot), marker='o', color='blue', label='Original')
    ax.plot(*zip(*vertices_init_np_plot), marker='*', color='red', label='s_init')
    ax.plot(*zip(*vertices_next_np_plot), marker='x', color='green', label='s_next')

    ax.plot(s_init[0], s_init[1], 'go', label='Initial State')
    ax.plot(s_next[0], s_next[1], 'ro', label='Next State')

    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    # assume the sample_time is 0.1, the velocity is 1.0, and the angular velocity is 0.5
    dt = 0.3
    w = 0.5
    v = 1.0
    s_next_2 = np.array([s_next[0] + dt * v * np.cos(s_next[2]),
                   s_next[1] + dt * v * np.sin(s_next[2]),
                   s_next[2] + dt * w])
    # get vertices for the next state s_next_2 or get A_next and b_next based on A_init and b_init
    theta_next_2 = s_next_2[2]
    p_next_2 = s_next_2[:2]
    R_next_2 = np.array([[np.cos(theta_next_2), -np.sin(theta_next_2)],
                    [np.sin(theta_next_2),  np.cos(theta_next_2)]])
    A_next_2 = A0 @ R_next_2.T
    b_next_2 = b0 + A_next_2 @ p_next_2
    vertices_next_2 = pypoman.compute_polygon_hull(A_next_2, b_next_2)
    vertices_next_2_np = np.array(vertices_next_2)
    vertices_next_2_np_plot = np.vstack((vertices_next_2_np, vertices_next_2_np[0]))  # close the polygon

    fig, ax = plt.subplots()
    ax.plot(*zip(*vertices_0_np_plot), marker='o', color='blue', label='Original')
    ax.plot(*zip(*vertices_init_np_plot), marker='*', color='red', label='s_init')
    ax.plot(*zip(*vertices_next_np_plot), marker='x', color='green', label='s_next')
    ax.plot(*zip(*vertices_next_2_np_plot), marker='x', color='purple', label='s_next_2')

    ax.plot(0, 0, 'o', label='Origin', color='blue')
    ax.plot(s_init[0], s_init[1], 'o', label='Initial State', color='red')
    ax.plot(s_next[0], s_next[1], 'o', label='Next State', color='green')
    ax.plot(s_next_2[0], s_next_2[1], 'o', label='Next State 2', color='purple')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    

if __name__ == "__main__":
    main()