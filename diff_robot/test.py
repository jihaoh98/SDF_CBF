import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
import numpy as np
from math import cos, sin, pi


class Test_Polytope:
    def __init__(self, center, width, height) -> None:
        self.center = np.array(center)
        self.width = width
        self.height = height

        self.vertexes = np.array([
            [self.center[0] - self.width, self.center[1] - self.height],
            [self.center[0] + self.width, self.center[1] - self.height],
            [self.center[0] + self.width, self.center[1] + self.height],
            [self.center[0] - self.width, self.center[1] + self.height]
        ])

        self.vertex_vectors = np.zeros_like(self.vertexes)
        for i in range(4):
            self.vertex_vectors[i, 0] = self.vertexes[i, 0] - self.center[0]
            self.vertex_vectors[i, 1] = self.vertexes[i, 1] - self.center[1]

    def get_vertexes(self, current_state):
        self.center = np.array(current_state)

        # first need to update the vertex_vector
        rotation_matrix = np.array(
            [[cos(self.center[2]), -sin(self.center[2])],
             [sin(self.center[2]), cos(self.center[2])]])
        
        # update the vertexes and the extended vertexes
        temp_vertexes_vector = (rotation_matrix @ self.vertex_vectors.T).T
        for i in range(4):
            self.vertexes[i, 0] = temp_vertexes_vector[i, 0] + self.center[0]
            self.vertexes[i, 1] = temp_vertexes_vector[i, 1] + self.center[1]

        return self.vertexes
    
    def rad_to_deg(self, rad):
        degrees = rad * (180 / pi)
        return degrees
    
    def plot(self):
        fig, ax = plt.subplots()

        # polytope1
        position = self.center[0:2] - np.array([self.width, self.height])
        polytope1 = mpatches.Rectangle((position[0], position[1]), self.width * 2, self.height * 2, edgecolor='r', facecolor='none')

        rotation_transform = transforms.Affine2D().rotate_around(self.center[0], self.center[1], self.center[2])
        polytope1.set_transform(rotation_transform + ax.transData)
        ax.add_patch(polytope1)

        polytope1.set_zorder(1)

        # polytope2
        polytope2 = mpatches.Polygon(self.vertexes, color='k')
        ax.add_patch(polytope2)
        polytope2.set_zorder(0)

        # 设置轴的界限
        ax.set_xlim(-2, 10)
        ax.set_ylim(-2, 10)
        ax.set_aspect('equal')

        # 显示图形
        plt.show()


if __name__ == "__main__":
    test = Test_Polytope([1.0, 1.0, 0.0], 1.0, 0.5)
    test.get_vertexes([2.0, 2.0, 0])
    test.plot()





