import numpy as np
import time
import yaml
import sys
import os
import statistics
import casadi as ca
from scipy.spatial import ConvexHull
from math import cos, sin
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import matplotlib.patches as mpatches
from scipy.integrate import solve_ivp
import time
import matplotlib.pyplot as plt


import casadi as ca

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



    def convex_polygon_hrep(self, points):
        """
        Given a set of 2D points (vertices of a convex polygon or a point cloud),
        compute the H-representation (A, b) of the convex polygon such that Ax ≤ b 
        describes the polygon (each inequality corresponds to one edge).
        """
        # Convert input to a NumPy array (n_points x 2)
        pts = np.asarray(points, dtype=float)
        if pts.shape[1] != 2:
            raise ValueError("Input points must be 2-dimensional coordinates.")
        
        # 1. Compute the convex hull of the points
        hull = ConvexHull(pts)
        
        # The ConvexHull vertices are in counterclockwise order (for 2D):contentReference[oaicite:6]{index=6}.
        # We could use hull.vertices (indices of hull points) if needed for further processing.
        # Here, we'll use hull.equations to get the facet equations directly.
        
        # 2. Get the hyperplane equations for each facet (edge) of the hull.
        # hull.equations is an array of shape (n_facets, 3) for 2D: [a, b, c] for each line (a*x + b*y + c = 0).
        # For interior points of the hull, a*x + b*y + c ≤ 0 holds true:contentReference[oaicite:7]{index=7}.
        equations = hull.equations  # shape (n_edges, 3)
        
        # 3. Split each equation into normal vector (a, b) and offset c.
        A = equations[:, :2]   # all rows, first two columns -> coefficients [a, b] for x and y
        c = equations[:, 2]    # last column is c in a*x + b*y + c = 0
        
        # 4. Convert to inequality form: a*x + b*y ≤ -c
        # We move c to the right side: a*x + b*y ≤ -c.
        b = -c  # Now each inequality is [a, b] · [x, y] ≤ b_i (where b_i = -c).
        
        # At this point, each row of A and corresponding element of b represent 
        # an inequality defining the half-space that contains the convex polygon.
        # (The normal vectors in A point outward, and the interior of the polygon 
        # satisfies A*x ≤ b.)

        b = b.reshape(-1, 1)  # Reshape b to be a column vector (n_edges x 1)
        
        return A, b, hull

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

def dyn_u(t, s, u, param):
    """
    Unicycle dynamics: s = [x, y, theta], u = [v, omega]
    Returns ds/dt as a 1D numpy array (shape (3,))
    """
    v, omega = u
    theta = s[2]
    head = param['head']

    dx = v * np.cos(theta + head)
    dy = v * np.sin(theta + head)
    dtheta = omega

    return np.array([dx, dy, dtheta])


def control(i, t_curr, s_curr, env, param):
    # Environment polytopes
    A = env['A']
    b = env['b']
    C = env['C']
    d = env['d']

    # Hyper-parameters
    a1 = param['a1']
    P = param['P']
    a2 = param['a2']
    eps_2 = param['eps_2']
    v_M = param['v_M']
    w_M = param['w_M']
    head = param['head']
    sf = param['sf']

    # Current position and orientation
    p = s_curr[0:2]  # x, y
    theta = s_curr[2]

    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    dR = np.array([
        [-np.sin(theta), -np.cos(theta)],
        [ np.cos(theta), -np.sin(theta)]
    ])

    # Indices for obstacle (mo_i) and robot (mr_i) polytopes
    mo_i = [0, 4, 8, 12]  # MATLAB: [1 5 9 13] → Python: 0-based
    mr_i = [0, 4, 8]      # MATLAB: [1 5 9] → Python: 0-based

    # Allocate dual variables and distances
    lmb = np.zeros((8, 6))      # Dual variables (4+4 per pair, 6 pairs)
    Ndist = np.zeros((6,))      # Distance values for 6 pairs

    log = {'time': np.zeros((1, 7)), 'duals': np.zeros((8, 6))}

    for o in range(3):
        for r in range(2):
            k = o*2 + r  # k = 0, 1, 2, 3, 4, 5

            Ai = A[mo_i[o]:mo_i[o] + 4, :]  # shape (4, 2)
            bi = b[mo_i[o]:mo_i[o] + 4, :]  # shape (4, 1), remember double check
            Ci = C[mr_i[r]:mr_i[r] + 4, :] @ R.T  # shape (4, 2)
            di = d[mr_i[r]:mr_i[r] + 4, :] + Ci @ p.reshape(-1, 1)  # shape (4, 1)

            opti = ca.Opti('conic')  # Create an optimization problem
            lam = opti.variable(8)  # Dual variables, the first four is for the robot and the last four is for the obstacle

            # Quadratic cost: -1/4 * [lam]^T H [lam] - f^T lam
            H_top = Ci @ Ci.T  # (4, 4)
            H_bot = Ai @ Ai.T  # (4, 4)
            H = 0.25 * ca.blockcat([[H_top, ca.MX.zeros(4, 4)],
                                    [ca.MX.zeros(4, 4), H_bot]])  # (8, 8)

            f = ca.vertcat(di, bi)  # shape (8, 1)
            cost =  (0.5 * ca.mtimes([lam.T, H, lam]) + ca.dot(f, lam))
            opti.minimize(cost)

            # Constraints
            Aeq = ca.horzcat(Ci.T, Ai.T)  # shape (2, 8)
            opti.subject_to(Aeq @ lam == 0)
            opti.subject_to(lam >= 0)

            # Solver options
            opts = {'print_time': 0, 'verbose': False}
            opti.solver('qpoases', opts)

            # Solve
            t0 = time.time()
            try:
                sol = opti.solve()
                lam_val = sol.value(lam)
                obj_val = sol.value(cost)
            except RuntimeError:
                lam_val = np.zeros((8,))
                obj_val = np.nan

            t1 = time.time()

            log['time'][0, 2 * o + r] = t1 - t0
            lmb[:, k] = lam_val
            Ndist[k] = -obj_val  # flip sign back: original maximization
    
    log['duals'] = lmb

    # Non-negativity constraints.
    temp = lmb.reshape((48, 1))  # shape (48, 1)
    Z = np.eye(48)
    mask = (temp < param['eps_2']).flatten()  # shape (48,)
    Z = -Z[mask, :]  # rows for which λ_k < eps_2

    # Step 2: Initialize constraint matrices
    Aeq = np.zeros((12, 2 + 48 + 1))  # equality: dual dynamics (6 pairs × 2D)
    beq = np.zeros((12, 1))


    n_Z_rows = Z.shape[0]
    n_ineq = 1 + 6 + n_Z_rows + 4 + 1  # CLF + 6 CBF + Z + input bounds + slack ≥ 0
    Ain = np.zeros((n_ineq, 2 + 48 + 1))  # inequality matrix
    bin = np.zeros((n_ineq, 1))

    # CLF constraint: first row of Ain
    delta_row = np.zeros((1, 2 + 48 + 1))

    # Compute heading terms
    direction = np.array([[np.cos(s_curr[2] + param['head'])],
                        [np.sin(s_curr[2] + param['head'])]])
    diff_pos = (s_curr[:2] - param['sf'][:2]).reshape(1, 2)
    theta_diff = s_curr[2] - param['sf'][2]

    # Fill in:
    # - dot product of position difference with direction (for u[0])
    # - angular difference (for u[1])
    delta_row[0, 0] = diff_pos @ direction  # affects v
    delta_row[0, 1] = theta_diff            # affects ω
    delta_row[0, -1] = -1                   # slack variable

    Ain[0, :] = delta_row
    bin[0, 0] = -param['a1'] * np.sum((s_curr - param['sf'])**2)

    # Slack constraint: last row of Ain
    Ain[-1, -1] = -1

    # n_Z_rows = Z.shape[0]
    row_start = 1 + 6 + n_Z_rows

    # Fill rows for input bounds: v, w ∈ [-v_M, v_M] × [-w_M, w_M]
    input_bound_block = np.vstack([
        np.eye(2),      # v ≤ v_M, ω ≤ w_M
        -np.eye(2)      # -v ≤ v_M, -ω ≤ w_M
    ])  # shape (4, 2)

    # Assign to Ain
    Ain[row_start:row_start + 4, 0:2] = input_bound_block
    bin[row_start:row_start + 4, 0] = np.array([v_M, w_M, v_M, w_M])

    # Construct block: [0_2 | Z | 0_1] ∈ ℝ^{m × 51}
    Z_block = np.hstack([
        np.zeros((n_Z_rows, 2)),       # zero block for u
        Z,                             # Z rows for selected dot(λ)
        np.zeros((n_Z_rows, 1))        # zero block for slack
    ])

    # Fill Ain with Z-based non-negativity constraints, bin entries remain 0 (already initialized)
    # Ain[1 + 6 : 1 + 6 + n_Z_rows, :] = Z_block

    for o in range(3):
        for r in range(2):
            k = 2 * o + r  # index from 0 to 5

            # Extract H-rep blocks
            Ai = A[mo_i[o]:mo_i[o]+4, :]  # (4×2)
            bi = b[mo_i[o]:mo_i[o]+4, :]

            Ci = C[mr_i[r]:mr_i[r]+4, :]  # (4×2)
            di = d[mr_i[r]:mr_i[r]+4, :] + Ci @ R.T @ p.reshape(-1, 1)

            l1 = lmb[0:4, k].reshape(-1, 1)  # for robot
            l2 = lmb[4:8, k].reshape(-1, 1)  # for obstacle

            # === Aeq: dual dynamics constraint ===
            row_eq = 2 * k
            Aeq[row_eq:row_eq+2, :] = np.hstack([
                np.zeros((2, 1)),                            # slack δ
                dR @ Ci.T @ l1,                              # ∂R/∂θ · Ci' · λ
                np.zeros((2, 8 * k)),                        # padding before dotλ
                R @ Ci.T,                                    # for control u
                Ai.T,                                        # for control u
                np.zeros((2, 8 * (5 - k) + 1))               # padding after
            ])

            # === Ain: CBF constraint ===
            row_ineq = 1 + k  # after CLF row (row 0)
            cbf_row = np.zeros((1, 2 + 48 + 1))


            # Compute scalar terms
            direction = np.array([[np.cos(s_curr[2] + param['head'])],
                                    [np.sin(s_curr[2] + param['head'])]])
            cbf_row[0, 0] = float(l1.T @ Ci @ R.T @ direction)       # affects v
            cbf_row[0, 1] = float(l1.T @ Ci @ dR.T @ p.reshape(-1,1))  # affects ω

            # Correct: 8 values into the right slice
            di_contrib = di.T                        # shape (1, 4)
            bi_contrib = bi.T + 0.5 * l2.T @ Ai @ Ai.T  # shape (1, 4)
            cbf_row[0, 2 + 8 * k : 2 + 8 * (k + 1)] = np.hstack([di_contrib, bi_contrib])

            Ain[row_ineq, :] = cbf_row
            bin[row_ineq, 0] = -2 * param['a2'] * (Ndist[k] + 0.015 ** 2)


    opti = ca.Opti()
    x = opti.variable(2 + 48 + 1)  # [v, w, dot_lambda(48), delta]
    H = np.eye(2 + 48 + 1)
    H[50, 50] = param['P']        # slack weight
    H = H + 1e-2 * np.eye(2 + 48 + 1)  # small regularization
    f = np.zeros((2 + 48 + 1,))  # linear term


    cost = 0.5 * ca.mtimes([x.T, H, x]) + ca.dot(f, x)
    opti.minimize(cost)

    # Stack all constraints
    A_total = np.vstack([Ain, Aeq])
    b_total = np.vstack([bin, beq])
    n_ineq = Ain.shape[0]
    n_eq = Aeq.shape[0]

    opti.subject_to(A_total[:n_ineq, :] @ x <= b_total[:n_ineq])
    opti.subject_to(A_total[n_ineq:, :] @ x == b_total[n_ineq:])

    x0 = np.zeros((2 + 48 + 1,))
    x0[-1] = 10.0  # delta init guess

    opti.set_initial(x, x0)

    # Solver settings
    p_opts = {"print_time": False}
    s_opts = {"print_level": 1}
    opti.solver("ipopt", p_opts, s_opts)

    start_time = time.time()
    try:
        sol = opti.solve()
        x_opt = sol.value(x)
    except RuntimeError as e:
        print("Solver failed:", e)
        x_opt = np.zeros((2 + 48 + 1,))

    log['time'][0, 6] = time.time() - start_time
    # Extract control input
    u = x_opt[0:2]  # [v, w]

    # Estimated h values (squared distance), negate to follow MATLAB logic
    h = -Ndist.copy()

    # dh/dt estimate: rows 1 to 6 (Python) == Ain[1:7, :] * x_opt
    dhdt_e = -Ain[1:7, :] @ x_opt  # shape (6,)
    # dhdt_e = dhdt_e.reshape(1, -1)  # shape (1, 6) like MATLAB's dhdt_e'

    return u, h, dhdt_e, log

def main():
    SHOW_SETUP = True

    rect_A = [[0.0, 0.0], [0.0, -1.0], [0.1, -1.0], [0.1, 0.0]]  # vertical part
    rect_B = [[0.0, 0.0], [0.0, -0.1], [1.0, -0.1], [1.0, 0.0]]  # horizontal part

    # define the robot
    robot = L_shaped_robot(
        indx=0,
        init_state=[0.05, 1.5, np.pi/4],  # move shared corner to (1.0, 1.0)
        rects=[rect_A, rect_B],
        mode='vertices',
        center_mode='vertex'
    )

    target_state = [7, 5.95, -np.pi/4]
    target_vertices = robot.get_vertices_at_absolute_state(target_state)
    

    # Hrep of the sofa robot
    mat_A, vec_a, _ = robot.convex_polygon_hrep(robot.vertices[0])
    mat_B, vec_b, _ = robot.convex_polygon_hrep(robot.vertices[1])
    
    # define the obstacle
    obs = [[[-2, 0.5], [0, 0.5], [0, 7], [-2, 7]], [[-2, 6], [8, 6], [8, 7],[-2, 7]], [[1, 0.5],[8, 0.5],[8, 5],[1, 5]]]
    mag_G = []
    vec_g = []
    for i in range(3):
        mat_G_i, vec_g_i, _ = robot.convex_polygon_hrep(obs[i])
        mag_G.append(mat_G_i)
        vec_g.append(vec_g_i)
    
    # plot to check the robot and obstacle setup
    if SHOW_SETUP:
        fig, ax = plt.subplots(figsize=(8, 8))
        color_list = ['blue', 'green']
        # # add robot
        for i in range(2):
            poly_robot = mpatches.Polygon(robot.vertices[i], alpha=0.5, color=color_list[i])
            ax.add_patch(poly_robot)

        # # add obs
        for i in range(3):
            poly_obs = mpatches.Polygon(obs[i], alpha=0.5, color='red')
            ax.add_patch(poly_obs)

        # plot the rotation center
        plt.scatter(robot.init_state[0], robot.init_state[1], c='black', marker='o', label='robot init')

        plt.axis('equal')
        plt.legend()
        plt.xlim(-2.5, 8.5)
        plt.ylim(-2.5, 8.5)
        plt.show()


    # define other parameters
    param = {
        'a1': 0.1,      # CLF constant [1/s]
        'a2': 1.0,      # CBF constant [1/s]
        'eps_2': 1e-5,  # U.S.C. parameter
        'v_M': 0.3,     # Velocity bound [m/s]
        'w_M': 0.2,     # Angular velocity bound [rad/s]
        'P': 10.0,       # CLF slack variable weight
        'head': np.pi/4,  # robot heading
        'sf': [7, 5.95, -np.pi/4]  # target state
    }

    s0 = [0.05, 1.5, np.pi/4]  # initial state

    env = {}
    rect = np.vstack([np.eye(2), -np.eye(2)])  # shape (4, 2)
    env['A'] = np.vstack([rect, rect, rect])  # shape (12, 2)
    env['b'] = np.array([
        0, 7, 2, -0.5,
        8, 7, 2, -6,
        8, 5, -1, -0.5
    ]).reshape(-1, 1)  # shape (12, 1)
    env['C'] = np.vstack([rect, rect])  # shape (8, 2)
    env['d'] = np.array([
        0.1, 0, 0, 0.9 + 0.1,
        0.9 + 0.1, 0, 0, 0.1
    ]).reshape(-1, 1)  # shape (8, 1)

    # # ==================================================================
    # Simulation parameters
    T = 30           # Total simulation time [s]
    dt = 0.05        # Time step [s]
    tspan = [0, T]   # Not used directly, just for reference

    # Time vector: 0 to T + dt, inclusive
    t = np.arange(0, T + dt, dt).reshape(-1, 1)  # shape (N+1, 1)
    N = t.shape[0]  # Number of time steps = len(t)

    # State trajectory: s ∈ ℝ^{N×3}, each row = [x, y, θ]
    s = np.zeros((N, 3))

    # Distance and derivative logs: for 6 wall-arm pairs
    h = np.zeros((N, 6))
    dhdt_e = np.zeros((N, 6))

    # Set initial state (s0 should be a 1D numpy array with 3 elements)
    s[0, :] = s0  # already Python-style indexing

    # Logging structure
    log = {
        'time': np.zeros((N - 1, 7)),     # Time per iteration
        'duals': np.zeros((8, 6, N - 1))  # Dual variables for each pair
    }

    display_text = ""


    # visualize the solution
    fig, ax = plt.subplots(figsize=(8, 8))
    color_list = ['blue', 'green']


    for i in range(N - 1):  # MATLAB: 1 to length(t)-1

        # Compute control input and dual values
        tic = time.time()
        u_t, h[i, :], dhdt_e[i, :], log_i = control(
            i=i,
            t_curr=dt * i,
            s_curr=s[i, :],  # already row vector
            env=env,
            param=param
        )
        loop_time = time.time() - tic

        # Print progress
        print('\r' + ' ' * len(display_text), end='')  # clear line
        display_text = f"time: {t[i,0]:.2f} s, loop time: {loop_time:.4f} s, frequency: {1/loop_time:.1f} Hz"
        print('\r' + display_text, end='')

        # Simulate forward using solve_ivp (like ode45)
        sol = solve_ivp(
            fun=lambda t_, s_: dyn_u(t_, s_, u_t, param),
            t_span=[0, dt],
            y0=s[i, :],
            method='RK45',
            t_eval=[dt]  # only want the result at the end of the step
        )
        s[i + 1, :] = sol.y[:, -1]  # store final state

        # Log data
        log['time'][i, :] = log_i['time']
        log['duals'][:, :, i] = log_i['duals']

        vertices_at_s_t =robot.get_vertices_at_absolute_state(s[i + 1, :])
        for i in range(2):
            poly_robot = mpatches.Polygon(vertices_at_s_t[i], alpha=0.5, color=color_list[i])
            ax.add_patch(poly_robot)

        # # add obs
        for i in range(3):
            poly_obs = mpatches.Polygon(obs[i], alpha=0.5, color='red')
            ax.add_patch(poly_obs)

        # # add robot
        for i in range(2):
            poly_robot = mpatches.Polygon(robot.vertices[i], alpha=0.5, color=color_list[i])
            ax.add_patch(poly_robot)

        plt.axis('equal')
        plt.legend()
        plt.xlim(-2.5, 8.5)
        plt.ylim(-2.5, 8.5)

        # plot the rotation center
        plt.scatter(s[i + 1, :][0], s[i + 1, :][1], c='black', marker='o', label='robot init')


        plt.show()




if __name__ == '__main__':
    main()