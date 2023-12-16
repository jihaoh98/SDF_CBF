import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def plot_squares(fixed_square, rotating_square):
    """
    Plot the two squares.
    :param fixed_square: Vertices of the fixed square.
    :param rotating_square: Vertices of the rotating square.
    """
    plt.figure(figsize=(6, 6))
    plt.plot(*(np.vstack((fixed_square, fixed_square[0, :]))).T, 'o-', label="Fixed Square")
    plt.plot(*(np.vstack((rotating_square, rotating_square[0, :]))).T, 'o-', label="Rotating Square")
    plt.grid(True)
    plt.xlim([-1, 4])
    plt.ylim([-1, 4])
    # plt.axis('equal')
    plt.legend()
    plt.show()


def create_square(center, size, angle):
    """
    Creates a square given a center, size, and rotation angle.
    """
    d = size / 2
    square = np.array([[-d, -d], [d, -d], [d, d], [-d, d]])
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    rotated_square = np.dot(square, rotation_matrix) + center
    return rotated_square


def closest_point(square, point):
    """
    Find the closest point on the square to a given point.
    """
    distances = np.linalg.norm(square - point, axis=1)
    closest_idx = np.argmin(distances)
    return square[closest_idx]


def jacobian_closest_point(square, center, size, angle, delta=0.01):
    """
    Calculate the Jacobian of the closest point on the square with respect to the rotation angle.
    """
    point_now = closest_point(square, fixed_square_center)
    square_delta = create_square(center, size, angle + delta)
    point_delta = closest_point(square_delta, fixed_square_center)
    jacobian = (point_delta - point_now) / delta
    return jacobian


def signed_distance(fixed_square, rotating_square, normal):
    """
    Compute the signed distance between two squares for a given normal vector.
    """
    proj_fixed = np.dot(fixed_square, normal)
    proj_rotating = np.dot(rotating_square, normal)
    min_distance = np.min(proj_rotating) - np.max(proj_fixed)
    return min_distance


def optimize_signed_distance(fixed_square, rotating_square):
    """
    Optimize the signed distance between two squares.
    """
    objective = lambda normal: -signed_distance(fixed_square, rotating_square, normal)
    initial_normal = np.array([1, 0])
    constraint = {'type': 'eq', 'fun': lambda normal: np.linalg.norm(normal) - 1}
    result = minimize(objective, initial_normal, constraints=constraint, method='SLSQP')
    max_signed_distance = -result.fun
    optimized_normal = result.x
    return max_signed_distance, optimized_normal


# Initialize squares and parameters
size = 1
fixed_square_center = (0, 0)
rotating_square_center = (2, 0)
initial_angle = np.deg2rad(45)

# Create squares
fixed_square = create_square(fixed_square_center, size, 0)
rotating_square = create_square(rotating_square_center, size, initial_angle)
plot_squares(fixed_square, rotating_square)

# Calculate the Jacobian and perform the linearization
jacobian_pA = jacobian_closest_point(rotating_square, rotating_square_center, size, initial_angle)
sd_initial, optimized_normal = optimize_signed_distance(fixed_square, rotating_square)
angle_delta = 0.1  # Small change in angle
linearized_sd = sd_initial + np.dot(optimized_normal, jacobian_pA) * angle_delta

print("Initial Signed Distance:", sd_initial)
print("Linearized Signed Distance:", linearized_sd)
print("Optimized Normal Vector:", optimized_normal)
