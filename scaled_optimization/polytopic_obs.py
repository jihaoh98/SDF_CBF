import numpy as np


class Polytopic_Obs:
    def __init__(self, indx, vertex, vel=np.zeros((2, )), goal=np.zeros((2, )), mode='static', **kwargs) -> None:
        """ init the polytopic-shaped obstacle """
        self.id = indx

        self.init_vertexes = np.array(vertex, ndmin=2)
        self.vertexes = np.copy(self.init_vertexes)
        self.ver_num = self.vertexes.shape[0]
        self.vertex_vectors = np.zeros_like(self.vertexes)

        self.vel = np.array(vel).reshape(2, )
        self.goal = np.array(goal).reshape(2, )
        self.mode = mode
        self.arrive_flag = False

        self.init_state = None
        self.cur_state = None
        self.A = None
        self.b = None
        self.b0 = None

        # init
        self.init()

    def init(self):
        self.get_center()
        self.get_vertex_vectors()
        self.get_initial_halfspace_constraints()

    def get_center(self):
        """ get the center position of the obstacle according the vertexes """
        center_x = 0.0
        center_y = 0.0

        for i in range(self.ver_num):
            center_x = center_x + self.init_vertexes[i, 0]
            center_y = center_y + self.init_vertexes[i, 1]

        center_x = center_x / self.ver_num
        center_y = center_y / self.ver_num

        # shape in (2, )
        self.init_state = np.array([center_x, center_y])
        self.cur_state = np.copy(self.init_state)

    def get_vertex_vectors(self):
        """ get the vertex vetcors """
        for i in range(self.ver_num):
            self.vertex_vectors[i, 0] = self.init_vertexes[i, 0] - self.init_state[0]
            self.vertex_vectors[i, 1] = self.init_vertexes[i, 1] - self.init_state[1]

    def get_initial_halfspace_constraints(self):
        """ get Ax <= b """
        A = []
        b = []

        for i in range(self.ver_num):
            p1 = self.vertexes[i]
            p2 = self.vertexes[(i + 1) % self.ver_num]

            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            normal = normal / np.linalg.norm(normal)

            b_i = -np.dot(normal, p1)
            A.append(-normal)
            b.append(b_i)

        self.A = np.array(A)
        self.b = np.array(b)
        self.b0 = self.b - np.dot(self.A, self.init_state[:2])

    def update_vertexes(self):
        """ update the vertexes when the obstacle moves, assume the obstacle only has translation """
        for i in range(self.ver_num):
            self.vertexes[i, 0] = self.cur_state[0] + self.vertex_vectors[i, 0]
            self.vertexes[i, 1] = self.cur_state[1] + self.vertex_vectors[i, 1]

    def get_current_vertexes(self, state):
        cur_vertexes = np.ones_like(self.vertex_vectors)
        for i in range(self.ver_num):
            cur_vertexes[i, 0] = state[0] + self.vertex_vectors[i, 0]
            cur_vertexes[i, 1] = state[1] + self.vertex_vectors[i, 1]

        return cur_vertexes

    def arrive_destination(self):
        """ determine if the obstacle arrives its goal position """
        dist = np.linalg.norm(self.cur_state - self.goal)

        if dist < 0.1:
            self.arrive_flag = True
            self.vel = np.zeros((2, ))
            return True
        else:
            self.arrive_flag = False
            return False

    def move_forward(self, step_time):
        """ move this obstacle """
        if self.mode != 'static':
            if self.arrive_flag:
                return
    
            self.cur_state = self.cur_state + self.vel * step_time
            if not self.arrive_flag:
                self.arrive_destination()

            self.update_vertexes()

    def get_current_state(self):
        """ return the obstacle's position and velocity """
        cur_state = np.array([self.cur_state[0], self.cur_state[1], self.vel[0], self.vel[1]])
        return cur_state
