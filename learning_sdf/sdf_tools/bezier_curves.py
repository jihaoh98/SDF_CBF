import numpy as np
import matplotlib.pyplot as plt
import scipy.special


def lerp(point_start, point_end, time):
    return point_start * (1 - time) + point_end * time


def decasteljau_bezier_curve(points, t):
    if len(points) == 1:
        return points[0]
    else:
        new_points = [lerp(points[i], points[i + 1], t) for i in range(len(points) - 1)]
        return decasteljau_bezier_curve(new_points, t)


def bernstein_poly(i, n, t):
    """
    The Bernstein polynomial of n, i as a function of t
    B_{i,n}(t) = (n,i) * t^i * (1-t)^(n-i)
    """
    return scipy.special.comb(n, i) * (t ** i) * (1 - t) ** (n - i)


def bernstein_bezier_curve(points, t):
    n = len(points) - 1
    curve_point = np.zeros(2)
    for i in range(n + 1):
        curve_point += scipy.special.comb(n, i) * ((1 - t) ** (n - i)) * (t ** i) * points[i]
    return curve_point


def polynomial_coefficients_bezier_curve(points, t):
    P0, P1, P2, P3 = points
    return P0 + t * (-3 * P0 + 3 * P1) + t ** 2 * (3 * P0 - 6 * P1 + 3 * P2) + t ** 3 * (-P0 + 3 * P1 - 3 * P2 + P3)


def my_binomial(n, k):
    if 0 <= k <= n:
        return scipy.special.comb(n, k)
    else:
        return 0


def generate_bezier_matrix(n):
    M = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            M[n - i, j] = (-1) ** (n - i - j) * my_binomial(n, j) * my_binomial(n - j, n - j - i)
    return M


def matrix_form_bezier_curve(points, t):
    n = len(points) - 1  # degree of the polynomial
    M = generate_bezier_matrix(n)
    T = np.array([t ** i for i in range(n + 1)])
    P = np.array(points)  # transpose to get points in columns

    dT = np.array([0 if i == 0 else i * t ** (i - 1) for i in range(n + 1)])  # derivative
    return T @ M @ P  # np.dot is used for matrix multiplication


def plot_straight_line(*points):
    for i in range(len(points) - 1):
        t_values = np.linspace(0, 1, 100)
        line_points = np.array([lerp(points[i], points[i + 1], t) for t in t_values])
        plt.plot(*line_points.T, color='k', linestyle='dashed')


def lerp_cubic_bezier_curves():
    # Define the control points
    P0 = np.array([0, 0])
    P1 = np.array([1, 2])
    P2 = np.array([2, 2])
    P3 = np.array([3, 0])
    # Generate the curve
    t_values = np.linspace(0, 1, 100)
    curve_points = np.array([decasteljau_bezier_curve([P0, P1, P2, P3], t) for t in t_values])

    fig, ax = plt.subplots()
    points = np.array([P0, P1, P2, P3])
    labels = ['P0', 'P1', 'P2', 'P3']
    plt.plot(*curve_points.T, color='red')  # curves
    plt.scatter(*points.T, color='blue', s=45, linewidths=2, facecolors='none')  # points
    plot_straight_line(P0, P1, P2, P3)
    for label, point in zip(labels, points):
        plt.text(point[0] + 0.05, point[1] + 0.05, label, fontsize=12)
    plt.title('Cubic Bezier Curve using De Casteljau\'s Algorithm')
    plt.axis('equal')
    plt.show()


def lerp_quadratic_bezier_curves():
    # Define the control points
    P0 = np.array([0, 0])
    P1 = np.array([1.5, 2])
    P2 = np.array([3, 0])

    # Generate the curve
    t_values = np.linspace(0, 1, 100)
    curve_points = np.array([decasteljau_bezier_curve([P0, P1, P2], t) for t in t_values])

    # Plot the curve and the control points
    fig, ax = plt.subplots()
    points = np.array([P0, P1, P2])
    labels = ['P0', 'P1', 'P2']
    plt.plot(*curve_points.T, color='red')
    plt.scatter(*points.T, color='blue', s=45, linewidths=2, facecolors='none')
    for label, point in zip(labels, points):
        plt.text(point[0] + 0.05, point[1], label, fontsize=12)
    plot_straight_line(P0, P1, P2)
    plt.title('Quadratic Bezier Curve using De Casteljau\'s Algorithm')
    plt.axis('equal')
    plt.show()


def bernstein_form_bezier_curves():
    # Define the control points
    P0 = np.array([0, 0])
    P1 = np.array([1, 2])
    P2 = np.array([2, 2])
    P3 = np.array([3, 0])
    P4 = np.array([2, -1])
    P5 = np.array([1, -1])

    # Generate the curve
    t_values = np.linspace(0, 1, 100)
    curve_points = np.array([bernstein_bezier_curve([P0, P1, P2, P3, P4, P5], t) for t in t_values])

    fig, ax = plt.subplots()
    points = np.array([P0, P1, P2, P3, P4, P5])
    labels = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5']
    plt.plot(*curve_points.T, color='red')  # curves
    plt.scatter(*points.T, color='blue', s=45, linewidths=2, facecolors='none')  # points
    plot_straight_line(P0, P1, P2, P3, P4, P5)
    for label, point in zip(labels, points):
        plt.text(point[0] + 0.05, point[1] + 0.05, label, fontsize=12)
    plt.title('N-Order Bezier Curve using Bernstein Polynomials')
    plt.axis('equal')
    plt.show()


def polynomial_form_bezier_curves():
    # Define the control points
    P0 = np.array([0, 0])
    P1 = np.array([1, 2])
    P2 = np.array([2, 2])
    P3 = np.array([3, 0])

    # Generate the curve
    t_values = np.linspace(0, 1, 100)
    curve_points = np.array([polynomial_coefficients_bezier_curve([P0, P1, P2, P3], t) for t in t_values])

    # Plot the curve and the control points
    points = np.array([P0, P1, P2, P3])
    labels = ['P0', 'P1', 'P2', 'P3']
    plt.plot(*curve_points.T, color='red')  # curves
    plt.scatter(*points.T, color='blue', s=45, linewidths=2, facecolors='none')  # points
    plot_straight_line(P0, P1, P2, P3)
    for label, point in zip(labels, points):
        plt.text(point[0] + 0.05, point[1] + 0.05, label, fontsize=12)
    plt.title('Cubic Bezier Curve using Polynomial Coefficients')
    plt.axis('equal')
    plt.show()


def matrix_form_bezier_curves():
    # Test
    P0 = np.array([0, 0])
    P1 = np.array([1, 2])
    P2 = np.array([2, 3])
    P3 = np.array([3, 2])
    P4 = np.array([5, 0])

    t_values = np.linspace(0, 1, 100)
    curve_points = np.array([matrix_form_bezier_curve([P0, P1, P2, P3, P4], t) for t in t_values])

    points = np.array([P0, P1, P2, P3, P4])
    labels = ['P0', 'P1', 'P2', 'P3', 'P4']
    plt.plot(*curve_points.T, color='red')  # curves
    plt.scatter(*points.T, color='blue', s=45, linewidths=2, facecolors='none')  # points
    plot_straight_line(P0, P1, P2, P3, P4)
    for label, point in zip(labels, points):
        plt.text(point[0] + 0.05, point[1] + 0.05, label, fontsize=12)
    plt.title('Cubic Bezier Curve using Polynomial Coefficients')
    plt.axis('equal')
    plt.show()

    n = points.shape[0] - 1
    plt.figure(figsize=(10, 6))
    for i in range(n + 1):
        B = bernstein_poly(i, n, t_values)
        plt.plot(t_values, B, label=f'P_{i},{n} = {scipy.special.comb(n, i)}*t^{i}*(1-t)^{n - i}')

    plt.legend()
    plt.xlabel('t')
    plt.ylabel('B_{i,n}(t)')
    plt.title(f'Bernstein Polynomials for a Bezier Curve of Degree {n}')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # lerp_cubic_bezier_curves()
    # lerp_quadratic_bezier_curves()
    # bernstein_form_bezier_curves()
    # polynomial_form_bezier_curves()
    matrix_form_bezier_curves()
