import numpy as np
from math import cos, sin


class Polytopic_robot:
    def __init__(self, indx, init_vertexes, step_time=0.1, goal=np.zeros((2, 1)), goal_margin=0.3, **kwargs) -> None:
        """ Init the polytopic robot """
        self.id = indx
        
        # vertexes given in （n, 2）, change to np.array (n, 2)
        if isinstance(init_vertexes, list):
            init_vertexes = np.array(init_vertexes, ndmin=2)

        if isinstance(goal, list):
            goal = np.array(goal, ndmin=2).T

        self.init_vertexes = init_vertexes
        self.vertexes = np.copy(self.init_vertexes)
        self.ver_num = self.vertexes.shape[0]
        self.vertex_vectors = np.zeros_like(self.vertexes)

        self.init_state = None
        self.cur_state = None
        
        self.step_time = step_time
        self.goal = goal
        self.goal_margin = goal_margin

        self.arrive_flag = False
        self.collision_flag = False
        self.init()

    def init(self):
        self.get_center()
        self.get_vertex_vectors()

    def get_center(self):
        """ get the center state """
        center_x = 0.0
        center_y = 0.0
        for i in range(self.ver_num):
            center_x = center_x + self.init_vertexes[i, 0]
            center_y = center_y + self.init_vertexes[i, 1]
        
        center_x = center_x / self.ver_num
        center_y = center_y / self.ver_num
        # TODO consider different theta
        center_theta = 0

        self.init_state = np.array([center_x, center_y, center_theta])
        self.cur_state = np.copy(self.init_state)

    def get_vertex_vectors(self):
        """ get the vertex vectors """
        for i in range(self.ver_num):
            self.vertex_vectors[i, 0] = self.vertexes[i, 0] - self.cur_state[0]
            self.vertex_vectors[i, 1] = self.vertexes[i, 1] - self.cur_state[1]

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

    def achieve_destination(self):
        pass

    def reset(self):
        pass
