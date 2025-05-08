import numpy as np
from math import cos, sin
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


class L_shaped_robot:
    def __init__(self, indx, model, init_vertices, step_time=0.1, goal=np.zeros((2, 1)), goal_margin=0.3, **kwargs) -> None:
        """ Init the l-shaped robot """
        self.id = indx
        self.model = model

        if isinstance(goal, list):
            goal = np.array(goal, ndmin=2).T

        self.init_vertices = init_vertices
        self.vertices = np.copy(self.init_vertices)

        # two vector of center point and length, width of L-shaped robot
        self.overlap_center = None
        self.center_vectors = None
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



    def get_bounds(self, vertices):
        vertices = np.array(vertices)
        x_min, x_max = vertices[:,0].min(), vertices[:,0].max()
        y_min, y_max = vertices[:,1].min(), vertices[:,1].max()
        return x_min, x_max, y_min, y_max
    
    def get_center(self, vertices):
        rect1 = vertices[0]
        rect2 = vertices[1]
        x1_min, x1_max, y1_min, y1_max = self.get_bounds(rect1)
        x2_min, x2_max, y2_min, y2_max = self.get_bounds(rect2)

        x_overlap_min = max(x1_min, x2_min)
        x_overlap_max = min(x1_max, x2_max)
        y_overlap_min = max(y1_min, y2_min)
        y_overlap_max = min(y1_max, y2_max)

        center_theta = 0

        if x_overlap_min < x_overlap_max and y_overlap_min < y_overlap_max:

            return (x_overlap_min + x_overlap_max)/2, (y_overlap_min + y_overlap_max)/2, center_theta
        else:
            return None  # no overlap


    def init_center(self):
        """ get the center state """
        center_x, center_y, center_theta = self.get_center(self.init_vertices)
        self.overlap_center = np.array([center_x, center_y])
        self.init_state = np.array([center_x, center_y, center_theta])
        self.cur_state = np.copy(self.init_state)


    def get_vertices(self, cur_state):
        vertices = np.zeros((2, 4, 2))
        for i in range(2):
            if i == 0:
                rotation_matrix = np.array([
                    [np.cos(cur_state[2]), -np.sin(cur_state[2])],
                    [np.sin(cur_state[2]), np.cos(cur_state[2])]
                ])
            elif i == 1:
                rotation_matrix = np.array([
                    [np.cos(cur_state[2]), -np.sin(cur_state[2])],
                    [np.sin(cur_state[2]), np.cos(cur_state[2])]
                ])
            
            translated_vertices = self.vertices[i] - self.overlap_center 
        
            rotated_vertices = np.dot(translated_vertices, rotation_matrix.T)  # Apply rotation
            rotated_vertices = rotated_vertices + self.overlap_center 
        
            translation_vector = cur_state[:2] - self.overlap_center
        
            cur_vertices = rotated_vertices + translation_vector
            vertices[i] = cur_vertices

        return vertices


    def update_vertexes(self):
        
        for i in range(2):
            if i == 0:
                rotation_matrix = np.array([
                    [np.cos(self.cur_state[2]), -np.sin(self.cur_state[2])],
                    [np.sin(self.cur_state[2]), np.cos(self.cur_state[2])]
                ])
            elif i == 1:
                rotation_matrix = np.array([
                    [np.cos(self.cur_state[2]), -np.sin(self.cur_state[2])],
                    [np.sin(self.cur_state[2]), np.cos(self.cur_state[2])]
                ])
            
            translated_vertices = self.vertices[i] - self.overlap_center 
        
            rotated_vertices = np.dot(translated_vertices, rotation_matrix.T)  # Apply rotation
            rotated_vertices = rotated_vertices + self.overlap_center 
        
            translation_vector = self.cur_state[:2] - self.overlap_center
        
            cur_vertices = rotated_vertices + translation_vector
            self.vertices[i] = cur_vertices


if __name__ == '__main__':
    rect_A = [[1,1], [1.1,1], [1.1,2], [1,2]]  # vertical
    rect_B = [[1,1], [2,1], [2,1.1], [1,1.1]]  # horizontal

    test_target = L_shaped_robot(0, None, [rect_A, rect_B])

    state_x, state_y, state_theta = test_target.get_center(test_target.init_vertices)

    # plot
    fig, ax = plt.subplots()
    poly_A = mpatches.Polygon(test_target.vertices[0], alpha=0.5, color='red')
    ax.add_patch(poly_A)
    poly_B = mpatches.Polygon(test_target.vertices[1], alpha=0.5, color='blue')
    ax.add_patch(poly_B)

    # plot the rotation center
    plt.scatter(test_target.cur_state[0], test_target.cur_state[1], c='black', marker='o', label='robot state')

    plt.axis('equal')
    plt.legend()
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    plt.show()

    fig, ax = plt.subplots()

    poly_A = mpatches.Polygon(test_target.vertices[0], alpha=0.5, color='red')
    ax.add_patch(poly_A)
    poly_B = mpatches.Polygon(test_target.vertices[1], alpha=0.5, color='blue')
    ax.add_patch(poly_B)
    plt.scatter(test_target.cur_state[0], test_target.cur_state[1], c='black', marker='o', label='init state')

    test_target.cur_state = np.array([2.0, 2.0, np.pi/6])
    test_target.update_vertexes()
    poly_A = mpatches.Polygon(test_target.vertices[0], alpha=0.5, color='red')
    ax.add_patch(poly_A)
    poly_B = mpatches.Polygon(test_target.vertices[1], alpha=0.5, color='blue')
    ax.add_patch(poly_B)

    # plot the rotation center
    plt.scatter(test_target.cur_state[0], test_target.cur_state[1], c='black', marker='o', label='cur state')

    plt.axis('equal')
    plt.legend()
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    plt.show()


    fig, ax = plt.subplots()

    poly_A = mpatches.Polygon(test_target.vertices[0], alpha=0.5, color='red')
    ax.add_patch(poly_A)
    poly_B = mpatches.Polygon(test_target.vertices[1], alpha=0.5, color='blue')
    ax.add_patch(poly_B)
    plt.scatter(test_target.cur_state[0], test_target.cur_state[1], c='black', marker='o', label='init state')

    xt = np.array([1.0, 1.0, np.pi/6])
    ver = test_target.get_vertices(xt)
    poly_A = mpatches.Polygon(ver[0], alpha=0.5, color='red')
    ax.add_patch(poly_A)
    poly_B = mpatches.Polygon(ver[1], alpha=0.5, color='blue')
    ax.add_patch(poly_B)
    plt.scatter(test_target.cur_state[0], test_target.cur_state[1], c='black', marker='o', label='cur state')


    # plot the rotation center

    plt.axis('equal')
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    plt.legend()
    plt.show()
