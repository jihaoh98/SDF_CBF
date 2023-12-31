import numpy as np
import matplotlib

# matplotlib.use('TkAgg')  # Do this BEFORE importing matplotlib.pyplotimport matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animation_moon(X, Y, ra, rb):
    fig = plt.figure()
    ax = plt.axes()

    def animate(i):
        ax.clear()
        t = i / 10  # vary t over time
        d = 0.25 + 0.45 * np.sin(t)  # make d vary with time

        distances = plot_sdMoon(X, Y, d, ra, rb)

        ax.contour(X, Y, distances, levels=50, cmap='RdGy')
        ax.contour(X, Y, distances, levels=[0], colors="red", linestyles='-', linewidths=3)

    ani = animation.FuncAnimation(fig, animate, frames=70, interval=10)
    plt.show()


def plot_sdMoon(X, Y, d, ra, rb):
    Y = np.abs(Y)
    a = (ra * ra - rb * rb + d * d) / (2.0 * d)
    b = np.sqrt(max(ra * ra - a * a, 0.0))
    conditions = d * (X * b - Y * a) > d * d * np.maximum(b - Y, 0.0)
    return np.where(conditions, np.linalg.norm(np.dstack([X - a, Y - b]), axis=-1),
                    np.maximum(np.linalg.norm(np.dstack([X, Y]), axis=-1) - ra,
                               -np.linalg.norm(np.dstack([X - d, Y]), axis=-1) + rb))


def sdMoon(p, d, ra, rb):
    p_y = np.abs(p[..., 1])
    a = (ra * ra - rb * rb + d * d) / (2.0 * d)
    b = np.sqrt(np.maximum(ra * ra - a * a, 0.0))
    conditions = d * (p[..., 0] * b - p_y * a) > d * d * np.maximum(b - p_y, 0.0)
    sdf = np.where(conditions, np.linalg.norm(p - np.array([a, b]), axis=-1),
                   np.maximum(np.linalg.norm(p, axis=-1) - ra,
                              -np.linalg.norm(p - np.array([d, 0]), axis=-1) + rb))
    inside = sdf < 0
    return sdf, inside


def gradient_moon(p, d, ra, rb, eps=1e-5):
    dx = np.array([eps, 0])
    dy = np.array([0, eps])
    grad_x = (sdMoon(p + dx, d, ra, rb)[0] - sdMoon(p - dx, d, ra, rb)[0]) / (2 * eps)
    grad_y = (sdMoon(p + dy, d, ra, rb)[0] - sdMoon(p - dy, d, ra, rb)[0]) / (2 * eps)
    return np.stack([grad_x, grad_y], axis=-1)


def project_to_boundary_moon(points, d, ra, rb):
    points_copy = np.copy(points)
    dists, insides = sdMoon(points_copy, d, ra, rb)
    grads = gradient_moon(points, d, ra, rb)
    projected_points = points - dists[:, np.newaxis] * grads / np.linalg.norm(grads, axis=-1, keepdims=True)
    projected_points = np.where(insides[:, np.newaxis], projected_points, points)
    return projected_points


def moon_shape():
    # Interpolate 100 points between A and B
    A = np.array([-1, -0.25])
    B = np.array([0.5, -0.25])
    points = np.linspace(A, B, 100)
    d = 0.25
    ra = 0.6
    rb = 0.5

    # fig, ax = plt.subplots()
    # Define the grid of points
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)

    # Plot projection
    # projected_points = project_to_boundary_moon(points, d=d, ra=ra, rb=rb)
    # plt.scatter(points[:, 0], points[:, 1], s=5, color='g')
    # plt.scatter(projected_points[:, 0], projected_points[:, 1], s=5, color='b')

    # plot distance field
    # distances = plot_sdMoon(X, Y, d, ra, rb)
    # plt.contour(X, Y, distances, levels=50, cmap='RdGy')
    # plt.contour(X, Y, distances, levels=[0], colors="red", linestyles='-', linewidths=3)

    # plot gradient field
    # P = np.dstack([X, Y])
    # grads = gradient_moon(P, d, ra, rb)
    # plt.streamplot(X, Y, grads[..., 0], grads[..., 1], density=1)

    # plot animation
    animation_moon(X, Y, ra, rb)
    plt.show()


def gradient_heart(points, eps=1e-5):
    """Calculate the gradient of the heart shape."""
    dx = np.array([eps, 0])
    dy = np.array([0, eps])
    grad_x = (sdHeart(points + dx)[0] - sdHeart(points - dx)[0]) / (2 * eps)
    grad_y = (sdHeart(points + dy)[0] - sdHeart(points - dy)[0]) / (2 * eps)
    return np.array([grad_x, grad_y]).T


def project_to_boundary_heart(points):
    """Project points inside the heart shape to its boundary."""
    points_copy = np.copy(points)
    dists, insides = sdHeart(points_copy)
    grads = gradient_heart(points)
    norm_grads = np.linalg.norm(grads, axis=1)
    projected_points = points - np.expand_dims(dists / norm_grads, axis=1) * grads
    projected_points = np.where(np.expand_dims(insides, axis=1), projected_points, points)
    return projected_points


def sdHeart(p, offset=np.array([0.0, -0.5])):
    """Calculate the signed distance function of the heart shape."""
    p = np.copy(p) - offset  # Subtract the offset from the points
    p[..., 0] = np.abs(p[..., 0])
    max_p = 0.5 * np.maximum(p[..., 0] + p[..., 1], 0.0)[..., np.newaxis]
    condition = p[..., 1] + p[..., 0] > 1.0
    sdf = np.where(condition,
                   np.sqrt(np.sum((p - np.array([0.25, 0.75])) ** 2, axis=-1)) - np.sqrt(2.0) / 4.0,
                   np.sqrt(np.minimum(np.sum((p - np.array([0.0, 1.0])) ** 2, axis=-1),
                                      np.sum((p - max_p) ** 2, axis=-1))) *
                   np.sign(p[..., 0] - p[..., 1]))
    inside = sdf < 0
    return sdf, inside


def heart_shape():
    """
    Visualize points interpolated between A and B,
    projected onto the boundary of a heart shape.
    Also plot the boundary of the heart shape.
    """
    A = np.array([-1, 0.35])
    B = np.array([1, 0.35])
    grid_dim = 1.5
    points = np.linspace(A, B, 100)

    fig, ax = plt.subplots()
    x = np.linspace(-grid_dim, grid_dim, 100)
    y = np.linspace(-grid_dim, grid_dim, 100)
    X, Y = np.meshgrid(x, y)
    P = np.dstack([X, Y])

    # plot projection
    # projected_points = project_to_boundary_heart(np.copy(points))
    # plt.scatter(points[:, 0], points[:, 1], s=5, color='b')
    # plt.scatter(projected_points[:, 0], projected_points[:, 1], s=5, color='k')

    # plot distance field
    distances, _ = sdHeart(P)
    plt.contour(X, Y, distances, levels=50, cmap='RdGy')
    plt.contour(X, Y, distances, levels=[0], colors="red", linestyles='-', linewidths=3)

    # plot gradient field
    # grads = gradient_heart(P.reshape(-1, 2)).reshape(100, 100, 2)
    # plt.streamplot(X, Y, grads[..., 0], grads[..., 1], density=1)
    plt.show()


def sdHorseshoe(p, c, r, w):
    p[..., 0] = np.abs(p[..., 0])
    l = np.linalg.norm(p, axis=-1)
    p = np.dstack([c[0] * p[..., 1] + c[1] * p[..., 0], c[1] * p[..., 1] - c[0] * p[..., 0]])
    p[..., 0] = np.where((p[..., 1] > 0) | (p[..., 0] > 0), p[..., 0], l * np.sign(-c[0]))
    p[..., 1] = np.where(p[..., 0] > 0, p[..., 1], l)
    p = np.dstack([p[..., 0], np.abs(p[..., 1] - r)]) - w
    dists = np.maximum(p[..., 0], p[..., 1])
    inside = dists < 0
    return dists, inside


def calculate_w(t):
    w = np.array([0.750, 0.25]) * (0.5 + 0.5 * np.cos(t * np.array([0.7, 1.1]) + np.array([0.0, 3.0])))
    return w


def calculate_c(t):
    c = np.array([np.cos(t), np.sin(t)])
    return c


def animation_horseshoe(X, Y):
    fig = plt.figure()
    ax = plt.axes()

    def animate(i):
        ax.clear()
        t = i / 10  # vary t over time
        w = calculate_w(t)
        c = calculate_c(t)
        r = 0.5

        P = np.dstack([X, Y])
        distances, _ = sdHorseshoe(P, c, r, w)

        ax.contour(X, Y, distances, levels=50, cmap='RdGy')
        ax.contour(X, Y, distances, levels=[0], colors="red", linestyles='-', linewidths=3)

    ani = animation.FuncAnimation(fig, animate, frames=100, interval=10)
    plt.show()


def gradient_horseshoe(p, c, r, w, original_x_negative=None, eps=1e-5):
    dx = np.array([eps, 0])
    dy = np.array([0, eps])
    grad_x = (sdHorseshoe(p + dx, c, r, w)[0] - sdHorseshoe(p - dx, c, r, w)[0]) / (2 * eps)
    grad_y = (sdHorseshoe(p + dy, c, r, w)[0] - sdHorseshoe(p - dy, c, r, w)[0]) / (2 * eps)
    grad_x = np.squeeze(grad_x)  # Ensure that grad_x is a 1D array
    grad_y = np.squeeze(grad_y)  # Ensure that grad_y is a 1D array
    # Negate the x-component of the gradient if the original x-coordinate was negative
    if original_x_negative is not None:
        grad_x = np.where(original_x_negative, -grad_x, grad_x)
    return np.stack([grad_x, grad_y], axis=-1)


def project_to_boundary_horseshoe(points, c, r, w):
    points_copy = np.copy(points)  # Create a copy of the points
    dists, insides = sdHorseshoe(points_copy, c, r, w)
    grads = gradient_horseshoe(points, c, r, w)
    grads_normalized = grads / np.linalg.norm(grads, axis=-1, keepdims=True)
    # Reshape dists and insides to match the shape of points
    dists = np.reshape(dists, (-1, 1))
    insides = np.reshape(insides, (-1, 1))
    # Project points using signed distances
    projected_points = points - dists * grads_normalized
    projected_points = np.where(insides, projected_points, points)
    return projected_points


def horseshoe_shape():
    # Define the parameters for the horseshoe
    t = 9.2
    w = calculate_w(t)
    c = calculate_c(t)
    r = 0.5
    # Define the grid of points
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)

    # Calculate the signed distance at every point on the grid
    P = np.dstack([X, Y])
    distances, _ = sdHorseshoe(P, c, r, w)

    # projection onto the surface
    A = np.array([-0.75, -0.25])
    B = np.array([0.75, -0.25])
    points = np.linspace(A, B, 100)
    projected_points = project_to_boundary_horseshoe(points, c, r, w)

    plt.figure(figsize=(6, 6))
    # plot distance field and zero level set
    plt.contour(X, Y, distances, levels=50, cmap='RdGy')
    plt.contour(X, Y, distances, levels=[0], colors="red", linestyles='-', linewidths=3)

    # plot projection
    # plt.scatter(points[:, 0], points[:, 1], s=8, color='g', zorder=5)  # plot original points
    # plt.scatter(projected_points[:, 0], projected_points[:, 1], s=8, color='b', zorder=5)  # plot projected points

    # plot gradient field
    # original_x_negative = X < 0
    # gradients = gradient_horseshoe(P, c, r, w, original_x_negative)
    # plt.streamplot(X, Y, gradients[..., 0], gradients[..., 1], density=1)

    # plot animation
    # animation_horseshoe(X, Y)
    plt.show()


if __name__ == "__main__":
    moon_shape()
    # heart_shape()
    # horseshoe_shape()
