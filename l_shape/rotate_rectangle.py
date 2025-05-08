import numpy as np
import matplotlib.pyplot as plt

# Define the original vertices of the rectangle
rect_A = np.array([[1, 1], [1.1, 1], [1.1, 2], [1, 2]])

# Define the initial center and target center (translation)
initial_center = np.array([1.05, 1.05])
target_center = np.array([2, 2])

# Define the rotation angle in degrees (30 degrees)
theta_deg = -30
theta_rad = np.radians(theta_deg)  # Convert to radians

# Function to rotate vertices around a center
def rotate(vertices, center, angle):
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    translated_vertices = vertices - center 
    rotated_vertices = np.dot(translated_vertices, rotation_matrix.T)  # Apply rotation
    return rotated_vertices + center 


def translate(vertices, translation_vector):
    return vertices + translation_vector

# Rotate the rectangle around its original center
rotated_vertices = rotate(rect_A, initial_center, theta_rad)

# Now translate the rotated rectangle to the target center
translation_vector = target_center - initial_center
final_vertices = translate(rotated_vertices, translation_vector)

# Plot the original and transformed rectangle
plt.figure()
plt.plot(rect_A[:, 0], rect_A[:, 1], label='Original Rectangle', color='blue', linestyle='-', marker='o')
plt.plot(rotated_vertices[:, 0], rotated_vertices[:, 1], label='Rotated Rectangle', color='green', linestyle='--', marker='x')
plt.plot(final_vertices[:, 0], final_vertices[:, 1], label='Final Transformed Rectangle', color='red', linestyle='-.', marker='^')
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.title("Rectangle Rotation and Translation")
plt.show()

# Print the new coordinates
print("Original vertices:\n", rect_A)
print("Rotated vertices:\n", rotated_vertices)
print("Final vertices after translation:\n", final_vertices)
