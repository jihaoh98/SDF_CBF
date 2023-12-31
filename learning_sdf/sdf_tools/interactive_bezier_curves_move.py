import numpy as np
import matplotlib.pyplot as plt
import scipy.special


def plot_bezier_curve(points):
    t_values = np.linspace(0, 1, 100)
    curve_points = np.array([matrix_form_bezier_curve(points, t) for t in t_values])
    plt.plot(*curve_points.T, color='red')


def on_press(event):
    global selected_point
    min_distance = float('inf')
    for i, point in enumerate(points):
        distance = np.linalg.norm(point - np.array([event.xdata, event.ydata]))
        if distance < min_distance:
            min_distance = distance
            selected_point = i
    redraw()


def on_release(event):
    global selected_point
    if selected_point is not None:
        points[selected_point] = np.array([event.xdata, event.ydata])
        selected_point = None  # Reset the selected point
    redraw()


def redraw():
    plt.gca().clear()  # Clear the current axes
    plt.xlim(-1, 10)
    plt.ylim(-1, 10)
    if len(points) > 0:
        plt.scatter(*np.array(points).T, color='blue')  # Draw points
    if len(points) > 1:
        plt.plot(*np.array(points).T, color='blue')  # Draw lines between points
    if len(points) > 2:
        plot_bezier_curve(points)  # Draw Bezier curve
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
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)

    # Draw initial points and Bezier curve
    redraw()

    plt.show()


# Define the global variable points
# points = [np.array([0, 0]), np.array([1, 2]), np.array([2, 4]), np.array([3, 5]), np.array([5, 6]), np.array([8, 0])]
# points = [np.array([0, 0]), np.array([4, 6]), np.array([8, 0])]
points = [np.array([0, 0]), np.array([3, 6]), np.array([5, 6]), np.array([8, 0])]

selected_point = None
if __name__ == '__main__':
    "Drag the points and observe the effect on curves"
    interactive_matri_form_bezier_curves()
