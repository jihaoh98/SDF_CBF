import numpy as np
import matplotlib.pyplot as plt
import scipy.special


def plot_bezier_curve(points):
    t_values = np.linspace(0, 1, 100)
    curve_points = np.array([matrix_form_bezier_curve(points, t) for t in t_values])
    plt.plot(*curve_points.T, color='red')


def on_click(event):
    global points
    points.append(np.array([event.xdata, event.ydata]))
    plt.scatter(*points[-1], color='blue')
    if len(points) > 2:
        plt.plot(*np.array(points).T, color='blue')
        plot_bezier_curve(points)
    elif len(points) == 2:
        plt.plot(*np.array(points).T, color='blue')
    plt.draw()


def on_release(event):
    global points
    points[-1] = np.array([event.xdata, event.ydata])
    while len(plt.gca().lines) > 0:
        plt.gca().lines[0].remove()
    plt.scatter(*np.array(points).T, color='blue')
    for i in range(len(points) - 1):
        plt.plot(*np.array(points[i:i + 2]).T, color='blue')
    if len(points) > 2:
        plot_bezier_curve(points)
    plt.draw()


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


def interactive_matri_form_bezier_curves():
    fig = plt.figure()
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('button_release_event', on_release)

    # Create an initial plot with a fixed range
    plt.xlim(-1, 4)
    plt.ylim(-1, 3)

    plt.show()


# Define the global variable points
points = []
if __name__ == '__main__':
    interactive_matri_form_bezier_curves()
