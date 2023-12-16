import numpy as np
from math import cos, sin
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


class L_shaped_robot:
    def __init__(self, indx, init_vertexes, step_time=0.1, goal=np.zeros((2, 1)), goal_margin=0.3, **kwargs) -> None:
        """ Init the l-shaped robot """
        self.id = indx

        # vertexes given in （n, 2）, change to np.array (n, 2)
        # start from the bottom left point
        if isinstance(init_vertexes, list):
            init_vertexes = np.array(init_vertexes, ndmin=2)

        if isinstance(goal, list):
            goal = np.array(goal, ndmin=2).T

        self.init_vertexes = init_vertexes
        self.vertexes = np.copy(self.init_vertexes)
        self.ver_num = self.vertexes.shape[0]
        self.vertex_vectors = np.zeros_like(self.vertexes)

        # two vector of center point and length, width of L-shaped robot
        self.center_vectors = None
        self.cur_center = None
        self.cur_center_body_frame = None
        self.width = None
        self.height = None

        self.init_state = None
        self.cur_state = None

        self.step_time = step_time
        self.goal = goal
        self.goal_margin = goal_margin

        self.arrive_flag = False
        self.collision_flag = False
        self.init()

    def init(self):
        self.init_center()
        self.init_vertex_vectors()
        self.init_two_center_vector()
        self.init_width_height()

    def get_center(self, vertexes):
        """ return the center of verteses """
        # vertexes in shape (N, 2) ndarray
        vertex_num = vertexes.shape[0]

        center_x = 0.0
        center_y = 0.0
        for i in range(vertex_num):
            center_x = center_x + vertexes[i, 0]
            center_y = center_y + vertexes[i, 1]

        center_x = center_x / vertex_num
        center_y = center_y / vertex_num
        # TODO consider different theta
        center_theta = 0

        return center_x, center_y, center_theta

    def init_center(self):
        """ get the center state """
        center_x, center_y, center_theta = self.get_center(self.init_vertexes)
        self.init_state = np.array([center_x, center_y, center_theta])
        self.cur_state = np.copy(self.init_state)

    def init_vertex_vectors(self):
        """ get the vertex vectors """
        for i in range(self.ver_num):
            self.vertex_vectors[i, 0] = self.vertexes[i, 0] - self.cur_state[0]
            self.vertex_vectors[i, 1] = self.vertexes[i, 1] - self.cur_state[1]

    def init_two_center_vector(self):
        """ init the center vector for two rectangles (combine it to L-shaped) """
        self.center_vectors = np.zeros((2, 2))
        self.cur_center = np.zeros((2, 2))

        # for the first center
        vertexes = np.array([
            [self.init_vertexes[0, 0], self.init_vertexes[0, 1]],
            [self.init_vertexes[1, 0], self.init_vertexes[1, 1]],
            [self.init_vertexes[2, 0], self.init_vertexes[2, 1]],
            [self.init_vertexes[0, 0], self.init_vertexes[2, 1]]
        ])
        self.cur_center[0, 0], self.cur_center[0, 1], _ = self.get_center(vertexes)

        # for the second center
        vertexes = np.array([
            [self.init_vertexes[0, 0], self.init_vertexes[0, 1]],
            [self.init_vertexes[4, 0], self.init_vertexes[0, 1]],
            [self.init_vertexes[4, 0], self.init_vertexes[4, 1]],
            [self.init_vertexes[5, 0], self.init_vertexes[5, 1]]
        ])
        self.cur_center[1, 0], self.cur_center[1, 1], _ = self.get_center(vertexes)

        for i in range(self.center_vectors.shape[0]):
            self.center_vectors[i, 0] = self.cur_center[i, 0] - self.cur_state[0]
            self.center_vectors[i, 1] = self.cur_center[i, 1] - self.cur_state[1]

        self.cur_center_body_frame = self.center_vectors

    def init_width_height(self):
        self.width = self.vertexes[1, 0] - self.vertexes[0, 0]
        self.height = self.vertexes[2, 1] - self.vertexes[1, 1]

    def get_vertexes(self, state):
        """ get the vertexes based on state """
        # update the vertex_vector
        rotation_matrix = np.array([
            [cos(state[2]), -sin(state[2])],
            [sin(state[2]), cos(state[2])]
        ])

        temp_vertex_vectors = (rotation_matrix @ self.vertex_vectors.T).T
        for i in range(self.ver_num):
            self.vertexes[i, 0] = state[0] + temp_vertex_vectors[i, 0]
            self.vertexes[i, 1] = state[1] + temp_vertex_vectors[i, 1]

        return self.vertexes

    def get_two_centers(self, state):
        """ get the two center of rectangles """
        rotation_matrix = np.array([
            [cos(state[2]), -sin(state[2])],
            [sin(state[2]), cos(state[2])]
        ])

        temp_vertex_vectors = (rotation_matrix @ self.center_vectors.T).T
        for i in range(temp_vertex_vectors.shape[0]):
            self.cur_center[i, 0] = state[0] + temp_vertex_vectors[i, 0]
            self.cur_center[i, 1] = state[1] + temp_vertex_vectors[i, 1]

        return self.cur_center

    def update_vertexes(self):
        """ update the vertexes """
        # update the vertex_vector
        rotation_matrix = np.array([
            [cos(self.cur_state[2]), -sin(self.cur_state[2])],
            [sin(self.cur_state[2]), cos(self.cur_state[2])]
        ])

        temp_vertex_vectors = (rotation_matrix @ self.vertex_vectors.T).T
        for i in range(self.ver_num):
            self.vertexes[i, 0] = self.cur_state[0] + temp_vertex_vectors[i, 0]
            self.vertexes[i, 1] = self.cur_state[1] + temp_vertex_vectors[i, 1]

    def arrive_destination(self):
        pass

    def reset(self):
        pass


if __name__ == '__main__':
    vertexes = [[1, 1], [2, 1], [2, 1.4], [1.4, 1.4], [1.4, 2], [1, 2]]
    test_target = L_shaped_robot(0, vertexes)

    # test
    print(test_target.cur_center)
    print(test_target.cur_state)
    print(test_target.cur_center_body_frame)
    print(test_target.length, test_target.width)

    new_state = np.array([1.8, 2.8, np.pi / 4])
    test_target.get_vertexes(new_state)
    test_target.get_two_centers(new_state)

    # plot
    fig, ax = plt.subplots()
    polytope = mpatches.Polygon(test_target.vertexes)

    ax.add_patch(polytope)
    # 设置坐标轴范围  

    ax.scatter(new_state[0], new_state[1], color='red', marker='o')
    ax.scatter(test_target.cur_center[0, 0], test_target.cur_center[0, 1], color='k', marker='o')
    ax.scatter(test_target.cur_center[1, 0], test_target.cur_center[1, 1], color='k', marker='o')
    ax.axis('equal')
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)

    plt.show()
