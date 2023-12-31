import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def filter_points(points, sdf, shape_params):
    filtered_points = []
    for point in points:
        # Compute the SDF at this point
        sdf_value = sdf(np.array(point).reshape(-1, 1), *shape_params)

        # If the SDF is greater than or equal to zero, this point is outside the shape
        if sdf_value >= -0.01:
            filtered_points.append(point)

    return np.array(filtered_points)


def sample_points_on_rectangle(center, width, length, num_samples):
    # Calculate the four corners of the rectangle
    corners = [
        [center[0] - width / 2, center[1] - length / 2],  # bottom left
        [center[0] + width / 2, center[1] - length / 2],  # bottom right
        [center[0] + width / 2, center[1] + length / 2],  # top right
        [center[0] - width / 2, center[1] + length / 2],  # top left
    ]
    points = []  # Sample points along the edges of the rectangle
    for i in range(4):
        # Calculate the direction vector for this edge
        dx, dy = np.subtract(corners[(i + 1) % 4], corners[i])

        # Generate evenly spaced numbers between 0 and 1
        t = np.linspace(0, 1, num_samples // 4)

        # Calculate the points along this edge
        edge_points = [[corners[i][0] + dx * tt, corners[i][1] + dy * tt] for tt in t]
        points.extend(edge_points)

    return np.array(points)


def sample_points_on_circle(center, radius, num_samples):
    thetas = np.linspace(0, 2 * np.pi, num_samples)
    x = center[0] + radius * np.cos(thetas)
    y = center[1] + radius * np.sin(thetas)
    return np.vstack([x, y]).T


def gradient(func, x, h=1e-5):
    x = np.array(x).reshape(-1, 1)
    dx = np.array([h, 0.0]).reshape(-1, 1)
    dy = np.array([0.0, h]).reshape(-1, 1)
    df_dx = (func(x + dx) - func(x - dx)) / (2 * h)
    df_dy = (func(x + dy) - func(x - dy)) / (2 * h)
    return np.array([df_dx[0, 0], df_dy[0, 0]])


def sdf(x, q=np.array([2, 0]), radius=0.6):  # generate circle with box
    length = radius
    width = radius / 2
    offset_q0 = 0.4
    offset_q1 = 0.6
    Mu1 = np.array([q[0], q[1]])  # circle center
    Mu2 = np.array([q[0] + offset_q0, q[1] + offset_q1])  # box center

    y = np.zeros((1, x.shape[1]))
    for t in range(x.shape[1]):
        "circle"
        d1 = (np.sqrt(np.dot((Mu1 - x[:, t]).T, (Mu1 - x[:, t]))) - radius)

        "box"
        dtmp = np.abs(x[:, t] - Mu2) - np.array([width, length])
        d2 = np.linalg.norm(np.maximum(dtmp, 0)) + np.minimum(np.maximum(dtmp[0], dtmp[1]), 0)
        k = 0.1
        h = np.maximum(k - np.abs(d1 - d2), 0)
        y[0, t] = np.minimum(d1, d2) - h ** 2 * 0.25 / k

    return y


def sdf_point_robot():
    """
    In practice, there is no need to sample points. The points are acquired by sensors.
    """
    accuracy = 100  # Number of datapoints per axis for visualization
    domain = 1.5  # for visualization
    X1, X2 = np.meshgrid(np.linspace(-domain, domain, accuracy), np.linspace(-domain, domain, accuracy))
    x = np.vstack((X1.T.ravel(), X2.T.ravel()))
    centter_q = np.array([0.0, 0.0])
    radii_q = 0.6
    y = sdf(x, centter_q, radii_q)  # compute the sdf
    e0 = 1E-6
    dx = np.zeros((x.shape[0], accuracy ** 2))
    for i in range(x.shape[0]):
        e = np.zeros((x.shape[0], 1))
        e[i] = e0
        ytmp = sdf(x + np.tile(e, (1, accuracy ** 2)))
        dx[i, :] = (y - ytmp) / e0

    # sample points from the sdf
    circle_points = sample_points_on_circle(centter_q, radii_q, 30)  # sample from the circle
    length = radii_q
    width = radii_q / 2
    offset_q0 = 0.4
    offset_q1 = 0.6
    Mu2 = np.array([centter_q[0] + offset_q0, centter_q[1] + offset_q1])  # compute the box center
    rectangle_points = sample_points_on_rectangle(Mu2, width * 2, length * 2, 30)
    all_points = np.vstack((circle_points, rectangle_points))  # Merge circle and rectangle points
    filtered_points = filter_points(all_points, sdf, [centter_q, radii_q])  # Filter points inside the shape
    # np.save('filtered_points.npy', filtered_points)

    # Plots
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    ax = fig.add_subplot(111, aspect='equal')
    contours = ax.contour(X1, X2, y.reshape(accuracy, accuracy).T, levels=15, linewidths=2)  # the distance field
    ax.contour(X1, X2, y.reshape(accuracy, accuracy).T, levels=[0], linewidths=2, colors='r')  # the shape of the robot
    plt.clabel(contours, inline=True, fontsize=12)
    plt.scatter(circle_points[:, 0], circle_points[:, 1])  # points on the circle
    plt.scatter(rectangle_points[:, 0], rectangle_points[:, 1])  # points on the box
    plt.scatter(filtered_points[:, 0], filtered_points[:, 1])  # points on the sdf surface
    ax.set_xlim([-domain, domain])
    ax.set_ylim([-domain, domain])
    red_patch = mpatches.Patch(color='red', label='Obstacle', fill=None, linewidth=2)  # add legend manually
    plt.legend(handles=[red_patch])
    plt.show()


if __name__ == '__main__':
    sdf_point_robot()
