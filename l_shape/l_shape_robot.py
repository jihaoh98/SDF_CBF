import numpy as np
from math import cos, sin
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


class L_shaped_robot:
    def __init__(self, indx, model=None, init_state=None, rects=None, size=None,
                 mode='size', center_mode='overlap', step_time=0.1, goal=np.zeros((2, 1)), goal_margin=0.3):
        self.id = indx
        self.model = model
        self.init_state = init_state
        self.step_time = step_time
        self.goal = goal
        self.goal_margin = goal_margin
        self.mode = mode
        self.center_mode = center_mode  # 'overlap' or 'vertex'
        self.current_state = None
        self.A_init = None
        self.a_init = None
        self.B_init = None
        self.b_init = None
        self.G_init = None
        self.g_init = None

        if mode == 'size':
            self.rect_length, self.rect_width = size
            self.init_vertices = self._build_L_shape_from_size_vertex()
        elif mode == 'vertices':
            if center_mode == 'vertex':
                self.init_vertices = self._normalize_to_vertex(rects)
            else:
                self.init_vertices = self._normalize_to_center(rects)
        else:
            raise ValueError("Mode must be either 'size' or 'vertices'")

        self.vertices = None
        self.init_vertices_consider_theta = None
        self.initialize_vertices()

    def _build_L_shape_from_size_vertex(self):
        """Build L-shape with shared vertex at origin (0,0)"""
        l, w = self.rect_length, self.rect_width

        # Vertical rectangle starts at (0, 0), goes up
        rect_A = [[0, 0], [w, 0], [w, l], [0, l]]

        # Horizontal rectangle starts at (0, 0), goes right
        rect_B = [[0, 0], [l, 0], [l, w], [0, w]]

        return [rect_A, rect_B]

    def _normalize_to_vertex(self, rects, tol=1e-6):
        """Shift so that the shared vertex of two rectangles is at origin"""
        shared = self._find_common_vertex(rects[0], rects[1], tol=tol)
        if shared is None:
            raise ValueError("No shared vertex found between rectangles")
        shifted = []
        for rect in rects:
            shifted.append([[x - shared[0], y - shared[1]] for (x, y) in rect])
        return shifted

    def _find_common_vertex(self, rect1, rect2, tol=1e-6):
        """Find a shared vertex between two rectangles (within tolerance)"""
        for v1 in rect1:
            for v2 in rect2:
                if np.linalg.norm(np.array(v1) - np.array(v2)) < tol:
                    return v1
        return None

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

    def get_h_form_rot_trans(self, s):
        p = s[:2].reshape(2, 1)
        theta = s[2]

        Rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        A_new = self.A_init @ Rotation_matrix.T
        a_new = self.a_init + self.A_init @ Rotation_matrix.T @ p
 
        B_new = self.B_init @ Rotation_matrix.T
        b_new = self.b_init + self.B_init @ Rotation_matrix.T @ p

        return A_new, a_new, B_new, b_new


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
        init_state=[0.05, 1.5, np.pi/4],  # move shared corner to (1.0, 1.0)
        rects=[rect_A, rect_B],
        mode='vertices',
        center_mode='vertex'
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

    absolute_move = [0.05, 1.5, np.pi/6]  # Move forward by 0.5 in robot's x direction and rotate 30 degrees more
    moved_vertices = robot.get_vertices_at_absolute_state(absolute_move)

    poly_A = mpatches.Polygon(moved_vertices[0], alpha=0.5, color='red')
    ax.add_patch(poly_A)
    poly_B = mpatches.Polygon(moved_vertices[1], alpha=0.5, color='blue')
    ax.add_patch(poly_B)
    plt.scatter(absolute_move[0], absolute_move[1], c='black', marker='o', label='robot init')

    plt.show()
