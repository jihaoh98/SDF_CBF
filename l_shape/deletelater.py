import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import HalfspaceIntersection
from scipy.spatial import ConvexHull

def convex_polygon_hrep(points):
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

A = np.vstack((np.eye(2), -np.eye(2)))
theta = np.pi/4
Rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])

A_new = A @ Rotation_matrix.T

b_0 = np.array([0.1, 0.0, 0.0, 1.0]).reshape(-1, 1)

b_new = b_0 + A @ Rotation_matrix.T @ np.array([0.05, 1.5]).reshape(-1, 1)

print(b_new)


vertices = [[0.0, 1.5], [0.7071067811865475, 0.7928932188134524], [0.7778174593052022, 0.8636038969321072], [0.07071067811865477, 1.5707106781186548]]
A_pkg, b_pkg, _ = convex_polygon_hrep(vertices)
print(b_pkg)