import numpy as np
from math import cos, sin
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


class L_shaped_robot:
    def __init__(self, indx, model=None, init_state=None, rects=None, size=None, mode='size', step_time=0.1, goal=np.zeros((2, 1)), goal_margin=0.3, **kwargs) -> None:
        self.id = indx
        self.goal = goal
        self.model = model
        self.goal_margin = goal_margin
        self.init_state = init_state
        self.step_time = step_time

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


        # two vector of center point and length, width of L-shaped robot
        self.overlap_center = None
        self.center_vectors = None
        self.init_state = init_state
        self.cur_state = None

        self.step_time = step_time
        self.goal = goal
        self.goal_margin = goal_margin

        self.arrive_flag = False
        self.collision_flag = False


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


    def get_vertices_at_relative_state(self, relative_state):
        """
        Get the transformed vertices if the robot moves from init_state by (dx, dy, dtheta).
        relative_state: [dx, dy, dtheta]
        """
        dx, dy, dtheta = relative_state
        x0, y0, theta0 = self.init_state

        # Rotate the delta position by theta0 (rotate in world frame)
        R0 = np.array([
            [np.cos(theta0), -np.sin(theta0)],
            [np.sin(theta0),  np.cos(theta0)]
        ])
        delta_pos = R0 @ np.array([dx, dy])

        # Final state
        new_x = x0 + delta_pos[0]
        new_y = y0 + delta_pos[1]
        new_theta = theta0 + dtheta

        # Apply this transformation to the shape
        transformed = []
        for rect in self.init_vertices:
            new_rect = [self._rotate_and_translate(pt, new_theta, new_x, new_y) for pt in rect]
            transformed.append(new_rect)

        return transformed


    def get_vertices_at_absolute_state(self, absolute_state):
        """Get the transformed vertices if robot is placed at absolute pose (x, y, theta)"""
        x, y, theta = absolute_state
        transformed = []
        for rect in self.init_vertices:
            new_rect = [self._rotate_and_translate(pt, theta, x, y) for pt in rect]
            transformed.append(new_rect)
        return transformed


if __name__ == '__main__':
    # robot = L_shaped_robot(
    # indx=0,
    # init_state=[1.0, 1.0, np.pi/6],
    # size=(1.0, 0.2),  # length=1.0, width=0.2
    # mode='size'
    # )
    # print(robot.vertices)

    # rect_A = [[0.0, 0.0], [0.1, 0.0], [0.1, 0.9], [0.0, 0.9]]   # vertical part
    # rect_B = [[0.0, 0.0], [0.9, 0.0], [0.9, 0.1], [0.0, 0.1]]   # horizontal part

    rect_A = [[0.0, 0.0], [0.0, -1.0], [0.1, -1.0], [0.1, 0.0]]  # vertical part
    rect_B = [[0.0, 0.0], [0.0, -0.1], [1.0, -0.1], [1.0, 0.0]]  # horizontal part
    

    robot = L_shaped_robot(
        indx=0,
        init_state=[0.0, 0.0, np.pi/4],  # pose refers to the center (will be auto-corrected by _normalize_to_center)
        rects=[rect_A, rect_B],
        mode='vertices'
    )   


    # plot
    fig, ax = plt.subplots()
    poly_A = mpatches.Polygon(robot.vertices[0], alpha=0.5, color='red')
    ax.add_patch(poly_A)
    poly_B = mpatches.Polygon(robot.vertices[1], alpha=0.5, color='blue')
    ax.add_patch(poly_B)

    # plot the rotation center
    plt.scatter(robot.init_state[0], robot.init_state[1], c='black', marker='o', label='robot init')

    plt.axis('equal')
    plt.legend()
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    relative_move = [0.5, 1.0, np.pi/6]  # Move forward by 0.5 in robot's x direction and rotate 30 degrees more
    moved_vertices = robot.get_vertices_at_absolute_state(relative_move)

    poly_A = mpatches.Polygon(moved_vertices[0], alpha=0.5, color='red')
    ax.add_patch(poly_A)
    poly_B = mpatches.Polygon(moved_vertices[1], alpha=0.5, color='blue')
    ax.add_patch(poly_B)
    plt.scatter(relative_move[0], relative_move[1], c='black', marker='o', label='robot init')

    plt.show()
