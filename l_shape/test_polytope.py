import numpy as np
import pypoman
import matplotlib.pyplot as plt

vertices = [[0, 0],[0, -0.1], [1, -0.1], [1, 0]]
vertices_np = np.array(vertices)
vertices_plot = np.array(vertices + ([vertices[0]]))  # close the polygon

A_init = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
b_init = np.array([0.5, 0.1, 0.5, 0.1])
p_init = np.array([0, 0])

vertices_init = pypoman.compute_polygon_hull(A_init, b_init)
vertices_init_np = np.array(vertices_init)
vertices_init_np_plot = np.vstack((vertices_init_np, vertices_init_np[0]))  # close the polygon

# translate and rotate
p0 = np.array([1.5, 1.5])
theta = np.pi/4
R = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])

vertices_rotated = np.dot(vertices_np, R.T) + p0
vertices_rotated_plot = np.vstack((vertices_rotated, vertices_rotated[0]))  # close the polygon

A_new = A_init @ R.T
b_new = b_init + A_new @ p0

vertices_new = pypoman.compute_polygon_hull(A_new, b_new)
vertices_new_np = np.array(vertices_new)
vertices_new_np_plot = np.vstack((vertices_new_np, vertices_new_np[0]))  # close the polygon
print('The vertices of the transformed polytope is ', vertices_new)




fig, ax = plt.subplots()
ax.plot(*zip(*vertices_init_np_plot), marker='o', color='blue', label='Original')
# ax.plot(*zip(*vertices_rotated_plot), marker='*', color='red', label='Transformed')



ax.set_aspect('equal')
ax.legend()
ax.set_xlim(-1, 3)
ax.set_ylim(-1, 3)
plt.grid()
plt.title('Transformed L-shape')
plt.show()