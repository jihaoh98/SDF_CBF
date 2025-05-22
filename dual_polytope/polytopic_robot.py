import numpy as np
from math import cos, sin
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


class polytopic_robot:
    def __init__(self, indx, model=None, init_state=None,  step_time=0.1, goal=np.zeros((2, 1)), goal_margin=0.3):
        self.id = indx
        self.model = model
        self.init_state = init_state
        self.step_time = step_time
        self.goal = goal
        self.goal_margin = goal_margin
        self.current_state = None
        self.A_init = None
        self.b_init = None
        self.G_init = None
        self.g_init = None

        self.vertices = None
        self.initialize()

    def initialize(self):
        self.current_state = np.array(self.init_state)

    
    def get_vertices_at_absolute_state(self, state):
        # state = [x, y, theta]
        x, y, theta = state
        R = np.array([[cos(theta), -sin(theta)],
                      [sin(theta), cos(theta)]])
        vertices_transformed = np.dot(self.vertices, R.T) + np.array([x, y])
        return vertices_transformed



# if __name__ == '__main__':
#     # robot = L_shaped_robot(
#     # indx=0,
#     # init_state=[1.0, 1.0, np.pi/6],
#     # size=(1.0, 0.2),  # length=1.0, width=0.2
#     # mode='size'
#     # )
#     # print(robot.vertices)

#     # rect_A = [[0.0, 0.0], [0.1, 0.0], [0.1, 0.9], [0.0, 0.9]]   # vertical part
#     # rect_B = [[0.0, 0.0], [0.9, 0.0], [0.9, 0.1], [0.0, 0.1]]   # horizontal part

#     rect_A = [[0.0, 0.0], [0.0, -1.0], [0.1, -1.0], [0.1, 0.0]]  # vertical part
#     rect_B = [[0.0, 0.0], [0.0, -0.1], [1.0, -0.1], [1.0, 0.0]]  # horizontal part
    
#     robot = L_shaped_robot(
#         indx=0,
#         init_state=[0.05, 1.5, np.pi/4],  # move shared corner to (1.0, 1.0)
#         rects=[rect_A, rect_B],
#         mode='vertices',
#         center_mode='vertex'
#     )


#     # plot
#     fig, ax = plt.subplots()
#     poly_A = mpatches.Polygon(robot.vertices[0], alpha=0.5, color='red')
#     ax.add_patch(poly_A)
#     poly_B = mpatches.Polygon(robot.vertices[1], alpha=0.5, color='blue')
#     ax.add_patch(poly_B)

#     # plot the rotation center
#     plt.scatter(robot.init_state[0], robot.init_state[1], c='black', marker='o', label='robot init')

#     plt.axis('equal')
#     plt.legend()
#     plt.xlim(-2, 2)
#     plt.ylim(-2, 2)

#     absolute_move = [0.05, 1.5, np.pi/6]  # Move forward by 0.5 in robot's x direction and rotate 30 degrees more
#     moved_vertices = robot.get_vertices_at_absolute_state(absolute_move)

#     poly_A = mpatches.Polygon(moved_vertices[0], alpha=0.5, color='red')
#     ax.add_patch(poly_A)
#     poly_B = mpatches.Polygon(moved_vertices[1], alpha=0.5, color='blue')
#     ax.add_patch(poly_B)
#     plt.scatter(absolute_move[0], absolute_move[1], c='black', marker='o', label='robot init')

#     plt.show()
