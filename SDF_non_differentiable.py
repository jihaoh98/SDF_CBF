import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def create_square(center, size, angle):
    """
    Creates a square given a center, size, and rotation angle.
    :param center: Tuple (x, y) for the center of the square.
    :param size: Length of the side of the square.
    :param angle: Rotation angle in radians.
    :return: Array of vertices of the square.
    """
    # Define the vertices of a square centered at origin
    d = size / 2
    square = np.array([[-d, -d], [d, -d], [d, d], [-d, d]])

    # Rotation matrix
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])

    # Rotate and translate the square
    rotated_square = np.dot(square, rotation_matrix) + center

    return rotated_square


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


def signed_distance(fixed_square, rotating_square, normal):
    """
    Compute the signed distance between two squares for a given normal vector.
    :param fixed_square: Vertices of the fixed square.
    :param rotating_square: Vertices of the rotating square.
    :param normal: Normal vector for projection.
    :return: Signed distance for the given normal vector.
    """
    # Project vertices onto the normal vector
    proj_fixed = np.dot(fixed_square, normal)
    proj_rotating = np.dot(rotating_square, normal)

    # Find the minimum distance between the projections
    min_distance = np.min(proj_rotating) - np.max(proj_fixed)

    return min_distance


def optimize_signed_distance(fixed_square, rotating_square):
    """
    Optimize the signed distance between two squares.
    :param fixed_square: Vertices of the fixed square.
    :param rotating_square: Vertices of the rotating square.
    :return: Maximum signed distance found by the optimization.
    """
    # Objective function: negative of the signed distance for maximization
    objective = lambda normal: -signed_distance(fixed_square, rotating_square, normal)

    # Initial guess for the normal vector
    initial_normal = np.array([1, 0])

    # Constraint: normal vector should be of unit length
    constraint = {'type': 'eq', 'fun': lambda normal: np.linalg.norm(normal) - 1}

    # Optimize to find the best normal vector
    result = minimize(objective, initial_normal, constraints=constraint, method='SLSQP')

    # Compute the maximum signed distance
    max_signed_distance = -result.fun

    return max_signed_distance, result.x


# Define two squares, one fixed and one that will rotate
size = 1  # Side length of the square
fixed_square_center = (1, 1)
rotating_square_center = (3, 1)
initial_angle = 0  # Initial angle in radians
# Create the fixed square
fixed_square = create_square(fixed_square_center, size, initial_angle)
# Create the rotating square with the initial angle
rotating_square = create_square(rotating_square_center, size, np.deg2rad(45))

# Example: Compute the signed distance for the initial pose
max_signed_distance, normal_vector = optimize_signed_distance(fixed_square, rotating_square)
print(max_signed_distance)
print(normal_vector)
proj_fixed = np.dot(fixed_square, normal_vector)
proj_rotating = np.dot(rotating_square, normal_vector)
max_index = np.argmax(proj_fixed)
min_index = np.argmin(proj_rotating)

# Plot both squares
plot_squares(fixed_square, rotating_square)


# Function to analyze the signed distance as a function of rotation angle
def analyze_signed_distance(fixed_square, center, size, angles):
    distances = []
    for angle in angles:
        # Create the rotating square at the current angle
        rotating_square = create_square(center, size, angle)

        # Compute the signed distance for this pose
        distance, _ = optimize_signed_distance(fixed_square, rotating_square)
        distances.append(distance)

    return distances


# Define a range of angles to analyze
angle_range = np.linspace(np.deg2rad(-45), np.deg2rad(45), 1000)

# Compute the signed distances for each angle
signed_distances = analyze_signed_distance(fixed_square, rotating_square_center, size, angle_range)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(angle_range, signed_distances, label="Signed Distance")
plt.xlabel("Rotation Angle (radians)")
plt.ylabel("Signed Distance")
plt.title("Signed Distance as a Function of Rotation Angle")
plt.legend()
plt.grid(True)
plt.show()
