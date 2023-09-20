# import numpy as np


# def calculate_sdf_rectangle_circle(rec_center, width, height, circle_center, radius):
#     """
#     This is a function which calcaulates the sdf between rectangle and circle
    
#     Args:
#         rectangle params: center, width, height (half of itself)
#         circle    params: center, radius

#     Returns:
#         sdf between rectangle and circle: float
#     """
#     dx = abs(circle_center[0] - rec_center[0]) - width
#     dy = abs(circle_center[1] - rec_center[1]) - height
    
#     distance_outside = np.linalg.norm([max(dx, 0), max(dy, 0)])
#     distance_inside = min(max(dx, dy), 0.0)
#     distance = distance_outside + distance_inside

#     sdf = distance - radius  
#     return sdf


# def calculate_sdf_relative(circle_relative_center, width, height, radius):
#     dx = abs(circle_relative_center[0]) - width
#     dy = abs(circle_relative_center[1]) - height
    
#     distance_outside = np.linalg.norm([max(dx, 0), max(dy, 0)])
#     distance_inside = min(max(dx, dy), 0.0)
#     distance = distance_outside + distance_inside

#     sdf = distance - radius  
#     return sdf


# def calculate_sdf_gradient_with_obstacle(rec_center, width, height, circle_center, radius):
#     """
#     This is a function which calcaulates the sdf gradient between rectangle and circle
    
#     Args:
#         rectangle params: center, width, height (half of itself)
#         circle    params: center, radius

#     Returns:
#         sdf gradient between rectangle and circle: np.array([x, y])
#     """
#     e0 = 1E-6
#     y = calculate_sdf_rectangle_circle(rec_center, width, height, circle_center, radius)
#     sdf_gradient = np.zeros((circle_center.shape[0], ))

#     for i in range(circle_center.shape[0]):
#         e = np.zeros((circle_center.shape[0], ))
#         e[i] = e0
#         ytmp = calculate_sdf_rectangle_circle(rec_center, width, height, circle_center + e, radius)
#         sdf_gradient[i] = (ytmp - y) / e0

#     return sdf_gradient


# def calculate_sdf_gradient_relative(circle_relative_center, width, height, radius):
#     e0 = 1E-6
#     y = calculate_sdf_relative(circle_relative_center, width, height, radius)
#     sdf_gradient = np.zeros((circle_relative_center.shape[0], ))

#     for i in range(circle_relative_center.shape[0]):
#         e = np.zeros((circle_relative_center.shape[0], ))
#         e[i] = e0
#         ytmp = calculate_sdf_relative(circle_relative_center + e, width, height, radius)
#         sdf_gradient[i] = (ytmp - y) / e0

#     return sdf_gradient


# rectangle_center = np.array([0.5, 1.0])
# rectangle_width = 1.0
# rectangle_height = 0.5

# # circle_center = np.array([2.0, 3.0])
# circle_radius = 0.5

# for i in np.arange(1.0, 3.0, 0.1):
#     for j in np.arange(1.5, 3.0, 0.1):
#         circle_center = np.array([i, j])
#         gradient1 = calculate_sdf_gradient_with_obstacle(rectangle_center, rectangle_width, rectangle_height, circle_center, circle_radius)
#         gradient2 = calculate_sdf_gradient_relative(circle_center - rectangle_center, rectangle_width, rectangle_height, circle_radius)
#         if gradient1.all() != gradient2.all():
#             print(i, j)

# # print()
# # print(calculate_sdf_gradient_with_obstacle(rectangle_center, rectangle_width, rectangle_height, circle_center, circle_radius))


import numpy as np

a = np.linspace(0, 1, 10)
print(a)
