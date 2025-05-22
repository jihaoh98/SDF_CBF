import numpy as np
from math import cos, sin
import cvxpy as cp
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


class Polytopic_Robot:
    def __init__(self, indx, init_vertexes, **kwargs) -> None:
        """ Init the polytopic robot """
        self.id = indx

        # vertexes given in （n, 2）, change to np.array (n, 2)
        if isinstance(init_vertexes, list):
            init_vertexes = np.array(init_vertexes, ndmin=2)

        self.init_vertexes = init_vertexes
        self.vertexes = np.copy(self.init_vertexes)
        self.ver_num = self.vertexes.shape[0]
        self.vertex_vectors = np.zeros_like(self.vertexes)
        
        self.init_state = None
        self.cur_state = None
        self.A = None
        self.b = None
        self.b0 = None
        self.G = None
        self.g = None
        self.vertices = None
    
        self.init()

    def init(self):
        self.get_center()
        self.get_vertex_vectors()
        # self.get_initial_halfspace_constraints()

    def get_center(self):
        """ get the center state """
        center_x = 0.0
        center_y = 0.0
        for i in range(self.ver_num):
            center_x = center_x + self.init_vertexes[i, 0]
            center_y = center_y + self.init_vertexes[i, 1]
        
        center_x = center_x / self.ver_num
        center_y = center_y / self.ver_num
        center_theta = 0

        self.init_state = np.array([center_x, center_y, center_theta])
        self.cur_state = np.copy(self.init_state)

    def get_vertex_vectors(self):
        """ get the vertex vectors """
        for i in range(self.ver_num):
            self.vertex_vectors[i, 0] = self.init_vertexes[i, 0] - self.init_state[0]
            self.vertex_vectors[i, 1] = self.init_vertexes[i, 1] - self.init_state[1]

    def update_state(self, state):
        """ update the state """
        self.cur_state = np.copy(state)
        self.update_vertexes()
    
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

    def get_vertexes(self, state):
        """ get the vertexes based on state """
        # update the vertex_vector
        rotation_matrix = np.array([
            [cos(state[2]), -sin(state[2])],
            [sin(state[2]), cos(state[2])]
        ])

        temp_vertex_vectors = (rotation_matrix @ self.vertex_vectors.T).T
        ans = np.zeros_like(self.vertexes)
        for i in range(self.ver_num):
            ans[i, 0] = state[0] + temp_vertex_vectors[i, 0]
            ans[i, 1] = state[1] + temp_vertex_vectors[i, 1]

        return ans

    def get_initial_halfspace_constraints(self):
        """ get the initial half-space constraints """
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

    def get_halfspace_constraints(self):
        """ get the half-space constraints """
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

        return np.array(A), np.array(b)
    
    def derive_min_distance(self, A1, b1, A2, b2):
        """ derive the minimum distance between two polytopes """
        x = cp.Variable((2, 1))
        y = cp.Variable((2, 1))

        # construct the optimization problem
        obj = cp.Minimize(cp.norm(y - x))
        cons = [A1 @ x <= b1.reshape(-1, 1), A2 @ y <= b2.reshape(-1, 1)]
        prob = cp.Problem(obj, cons)

        # solve the problem
        optimal_value = prob.solve()
        print('optimal_value:', optimal_value)

    def transform_verify(self, t):
        """ verify the transformation """
        R = np.array(
            [[np.cos(t[2]), -np.sin(t[2])], 
            [np.sin(t[2]), np.cos(t[2])]]
        )
        A1 = (R @ self.A.T).T
        b1 = self.b0 + np.dot(A1, t[0:2]) 
        print(A1)
        print(b1)

        self.update_state(t)
        A2, b2 = self.get_halfspace_constraints()
        print('After update:')
        print(A2)
        print(b2)

    def derive_vertices(self, A, b):
        """ derive the vertices """
        num_constraints = A.shape[0]
        vertices = []

        for i in range(num_constraints):
            A_sub = np.array([A[i], A[(i + 1) % num_constraints]]) 
            b_sub = np.array([b[i], b[(i + 1) % num_constraints]])

            if np.linalg.matrix_rank(A_sub) == 2:
                intersection = np.linalg.solve(A_sub, b_sub)
                vertices.append(intersection)

        vertices = np.array(vertices)
        if len(vertices) > 2:
            hull = ConvexHull(vertices)
            vertices = vertices[hull.vertices]

        return vertices
        
    def plot_verify(self):
        """ plot the robot to verify  the scale """
        fig, ax = plt.subplots()

        t = [self.init_state[0], self.init_state[1], 0.0]
        R = np.array(
            [[np.cos(t[2]), -np.sin(t[2])], 
            [np.sin(t[2]), np.cos(t[2])]]
        )
        A1 = (R @ self.A.T).T
        b1 = self.b0 + np.dot(A1, t[0:2]) 
        vertices1 = self.derive_vertices(A1, b1)

        b2 = 4 * self.b0 + np.dot(A1, t[0:2])
        vertices2 = self.derive_vertices(A1, b2)

        for i in range(len(vertices1)):
            p1 = vertices1[i]
            p2 = vertices1[(i + 1) % len(vertices1)]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-')

        for i in range(len(vertices2)):
            p1 = vertices2[i]
            p2 = vertices2[(i + 1) % len(vertices2)]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-')

        plt.show()


if __name__ == "__main__":
    vertices = ([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 2.0]])
    robot = Polytopic_Robot(0, vertices)
    robot.plot_verify()

    # t = [1, 2.5, np.pi / 4]
    # robot.transform_verify(t)
