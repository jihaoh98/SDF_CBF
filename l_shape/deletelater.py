import numpy as np
import numpy as np
from math import cos, sin
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

class L_shaped_robot:
    def __init__(self, indx, model=None, init_state=None, rects=None, size=None, mode='size', step_time=0.1, goal=np.zeros((2, 1)), goal_margin=0.3):
        self.id = indx
        self.model = model
        self.init_state = init_state
        self.step_time = step_time
        self.goal = goal
        self.goal_margin = goal_margin

        self.mode = mode
        if mode == 'size':
            self.rect_length, self.rect_width = size
            self.init_vertices = self._build_L_shape_from_size()
        elif mode == 'vertices':
            self.init_vertices = self._normalize_to_center(rects)
        else:
            raise ValueError("Mode must be either 'size' or 'vertices'")

        self.vertices = None
        self.init_vertices_consider_theta = None
        self.initialize_vertices()

    def _build_L_shape_from_size(self):
        l, w = self.rect_length, self.rect_width

        # Build vertical rectangle centered at origin intersection
        rect_A = [[-w/2, 0], [w/2, 0], [w/2, l], [-w/2, l]]

        # Build horizontal rectangle centered at origin intersection
        rect_B = [[0, -w/2], [l, -w/2], [l, w/2], [0, w/2]]

        return [rect_A, rect_B]

    def _normalize_to_center(self, rects):
        """Shift given rectangles so their intersection center is at origin"""
        center = self.get_center(rects)
        if center is None:
            raise ValueError("Provided rectangles do not overlap")
        cx, cy, _ = center
        shifted = []
        for rect in rects:
            shifted.append([[x - cx, y - cy] for (x, y) in rect])
        return shifted

    def initialize_vertices(self):
        """Rotate and translate L-shape according to init_state"""
        x, y, theta = self.init_state
        transformed = []
        for rect in self.init_vertices:
            new_rect = [self._rotate_and_translate(pt, theta, x, y) for pt in rect]
            transformed.append(new_rect)
        self.vertices = transformed
        self.init_vertices_consider_theta = transformed

    def _rotate_and_translate(self, pt, theta, dx, dy):
        """Rotate point around origin and translate"""
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        pt = np.array(pt)
        return (R @ pt + np.array([dx, dy])).tolist()

    def get_bounds(self, vertices):
        vertices = np.array(vertices)
        x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
        y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
        return x_min, x_max, y_min, y_max

    def get_center(self, rects):
        rect1 = rects[0]
        rect2 = rects[1]
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
            return None



def main():
    robot = L_shaped_robot(
    indx=0,
    init_state=[1.0, 1.0, np.pi/6],
    size=(1.0, 0.2),  # length=1.0, width=0.2
    mode='size'
    )
    print(robot.vertices)


    rect_A = [[1, 1], [1.1, 1], [1.1, 2], [1, 2]]
    rect_B = [[1, 1], [2, 1], [2, 1.1], [1, 1.1]]

    robot = L_shaped_robot(
        indx=0,
        init_state=[0.0, 0.0, 0],
        rects=[rect_A, rect_B],
        mode='vertices'
    )
    print(robot.vertices)

    fig, ax = plt.subplots()
    poly_A = mpatches.Polygon(robot.vertices[0], alpha=0.5, color='red')
    ax.add_patch(poly_A)
    poly_B = mpatches.Polygon(robot.vertices[1], alpha=0.5, color='blue')
    ax.add_patch(poly_B)


    plt.axis('equal')
    plt.legend()
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    plt.show()



if __name__ =='__main__':
    main()