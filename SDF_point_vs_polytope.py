import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cvxpy as cp


def rotate_point(p, theta):
    """Rotate a point by theta degrees."""
    theta = np.radians(theta)  # Convert to radians
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    return rotation_matrix.dot(p)


def distance_to_rotated_rectangle(px, py, rx, ry, rectangle_center, rotation_angle):
    """
    Compute the shortest distance from a point to a rotated and translated rectangle.
    This version correctly considers all cases, including when the point is inside the rectangle.
    """
    # Transform the point relative to the rectangle
    transformed_point = rotate_point(np.array([px, py]) - rectangle_center, -rotation_angle)

    # Distance calculation considering if the point is inside the rectangle
    dx = max(abs(transformed_point[0]) - rx, 0)
    dy = max(abs(transformed_point[1]) - ry, 0)
    distance = np.sqrt(dx ** 2 + dy ** 2)
    return distance


def plot_scenario(px, py, rx, ry, rectangle_center, rotation_angle):
    """Plot the point, rectangle, and distance."""
    fig, ax = plt.subplots()
    # Create the rectangle
    rectangle = patches.Rectangle((-rx, -ry), 2 * rx, 2 * ry, linewidth=1, edgecolor='r', facecolor='none')

    # Transform the rectangle
    t = patches.transforms.Affine2D().rotate_deg_around(0, 0, rotation_angle) + patches.transforms.Affine2D().translate(
        *rectangle_center) + ax.transData
    rectangle.set_transform(t)
    ax.add_patch(rectangle)

    # Plot the point
    ax.plot(px, py, 'bo')  # Original point

    # Set limits and labels
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Point and Rotated, Translated Rectangle')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')

    plt.show()


# Example parameters
# px, py = 2., 4.0001  # Point coordinates
px, py = 6, 6.0001  # Point coordinates
rx, ry = 2., 2.  # Half-width and half-height of the rectangle
rectangle_center = (0, 0)  # Translation of the rectangle
rotation_angle = 0  # Rotation angle in degrees

# Compute the distance
distance = distance_to_rotated_rectangle(px, py, rx, ry, rectangle_center, rotation_angle)
print(f"The shortest distance: {distance}")

# Plot the scenario
plot_scenario(px, py, rx, ry, rectangle_center, rotation_angle)
# Corrected debugging for rotation
distances_corrected = []
rotations = np.linspace(-125, 125, 10000)

for angle in rotations:
    distance = distance_to_rotated_rectangle(px, py, rx, ry, rectangle_center, angle)
    distances_corrected.append(distance)

print(f"Minimum distance: {min(distances_corrected)}")
# Plotting the corrected results
plt.figure(figsize=(6, 4))
plt.plot(rotations, distances_corrected)
plt.title('Corrected Distance vs Rotation')
plt.xlabel('Rotation Angle (degrees)')
plt.ylabel('Distance')
plt.show()

"""
As long as the robot is not extremely close to the obstacle, the sdf distance is differentiable.
"""
