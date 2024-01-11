import numpy as np
import matplotlib.pyplot as plt


def polySpline(p, AA, BB, CC, FF, GG, HH):
    odd = False
    eps = 1e-7
    X = p[0]
    Y = p[1]

    def setX(f, a, b):
        nonlocal odd
        if f >= 0. and f < 1. and np.where(f <= 0.5, a[0], b[0]) < X:
            odd = not odd

    a = AA
    b = BB
    c = CC
    A = a[1] + c[1] - b[1] - b[1]
    B = 2. * (b[1] - a[1])
    C = a[1] - Y
    if np.abs(A) < eps:
        setX(-C / B, a, b)
    else:
        root = B * B - 4. * A * C
        if root > 0.:
            root = np.sqrt(root)
            setX((-B - root) / (2. * A), a, b)
            setX((-B + root) / (2. * A), a, b)
    a = c

    b = FF
    c = GG
    A = a[1] + c[1] - b[1] - b[1]
    B = 2. * (b[1] - a[1])
    C = a[1] - Y
    if np.abs(A) < eps:
        setX(-C / B, a, b)
    else:
        root = B * B - 4. * A * C
        if root > 0.:
            root = np.sqrt(root)
            setX((-B - root) / (2. * A), a, b)
            setX((-B + root) / (2. * A), a, b)
    a = c

    b = HH
    c = AA
    A = a[1] + c[1] - b[1] - b[1]
    B = 2. * (b[1] - a[1])
    C = a[1] - Y
    if np.abs(A) < eps:
        setX(-C / B, a, b)
    else:
        root = B * B - 4. * A * C
        if root > 0.:
            root = np.sqrt(root)
            setX((-B - root) / (2. * A), a, b)
            setX((-B + root) / (2. * A), a, b)

    return float(odd)


def lerp(x, y, a):
    return x * (1 - a) + y * a


def step(edge, x):
    return np.where(x < edge, 0.0, 1.0)


def solveCubic(a, b, c):
    coefficients = [1, a, b, c]
    roots = np.roots(coefficients)
    roots = np.real(roots[np.isreal(roots)])  # only consider real roots
    roots = np.clip(roots, 0, 1)  # clamp values between 0 and 1
    return roots


def sdBezier(A, B, C, p):
    eps = 1e-4
    B = np.where(np.abs(np.sign(B * 2.0 - A - C)) < eps, B + np.array([eps, eps]), B)
    a = B - A
    b = A - B * 2.0 + C
    c = a * 2.0
    d = A - p

    k = np.array([3. * np.dot(a, b), 2. * np.dot(a, a) + np.dot(d, b), np.dot(d, a)]) / np.dot(b, b)
    t = solveCubic(k[0], k[1], k[2])

    pos = A + (c + b * t[0]) * t[0]
    dis = np.linalg.norm(pos - p)
    for i in range(1, len(t)):
        pos = A + (c + b * t[i]) * t[i]
        dis = min(dis, np.linalg.norm(pos - p))
    return dis


def testCross(a, b, p):
    return np.sign((b[1] - a[1]) * (p[0] - a[0]) - (b[0] - a[0]) * (p[1] - a[1]))


def signBezier(A, B, C, p):
    a = C - A
    b = B - A
    c = p - A
    bary = np.array([c[0] * b[1] - b[0] * c[1], a[0] * c[1] - c[0] * a[1]]) / (a[0] * b[1] - b[0] * a[1])
    d = np.array([bary[1] * 0.5, 0.0]) + 1.0 - bary[0] - bary[1]
    return lerp(np.sign(d[0] * d[0] - d[1]), lerp(-1.0, 1.0,
                                                  step(testCross(A, B, p) * testCross(B, C, p), 0.0)),
                step((d[0] - d[1]), 0.0)) * testCross(A, C, B)


def plot_points(A, B, C, F, G, H, I, J):
    plt.scatter(*A, color='red', s=100, label='A')
    plt.scatter(*B, color='green', label='B')
    plt.scatter(*C, color='blue', s=100, label='C')
    plt.scatter(*F, color='pink', label='F')
    plt.scatter(*G, color='purple', s=100, label='G')
    plt.scatter(*H, color='orange', label='H')
    plt.scatter(*I, color='pink', s=100, label='I')
    plt.scatter(*J, color='k', label='J')


def bezier_curve(t, A, B, C):
    return (1 - t) ** 2 * A + 2 * (1 - t) * t * B + t ** 2 * C


def plot_curves(A, B, C, F, G, H, I, J):
    t = np.linspace(0, 1, 100).reshape(-1, 1)
    curve1 = bezier_curve(t, A, B, C)
    curve2 = bezier_curve(t, C, F, G)
    curve3 = bezier_curve(t, G, H, I)
    curve4 = bezier_curve(t, I, J, A)

    plt.plot(*curve1.T, color='red', linewidth=4)
    plt.plot(*curve2.T, color='green', linewidth=4)
    plt.plot(*curve3.T, color='blue', linewidth=4)
    plt.plot(*curve4.T, color='purple', linewidth=4)


def contour_poly_spline(A, B, C, F, G, H, I, J):
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            p = np.array([X[i, j], Y[i, j]])
            d1 = sdBezier(A, B, C, p)
            d2 = sdBezier(C, F, G, p)
            d3 = sdBezier(G, H, I, p)
            d4 = sdBezier(I, J, A, p)

            Z[i, j] = min(d1, d2, d3, d4)

    plt.figure(figsize=(12, 12))
    plt.contour(X, Y, Z, levels=100, cmap='RdGy')
    # plt.contour(X, Y, Z, levels=[0], colors='r')  # adjust levels and colormap as necessary
    plot_curves(A, B, C, F, G, H, I, J)
    plot_points(A, B, C, F, G, H, I, J)
    plt.legend()
    plt.show()


# Define the function to plot the gradient field
def plot_gradient_field(A, B, C, F, G, H, I, J):
    # Define the grid of points
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    # Compute the distance field
    Z = np.zeros_like(X)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            p = np.array([X[i, j], Y[i, j]])
            d1 = sdBezier(A, B, C, p)
            d2 = sdBezier(C, F, G, p)
            d3 = sdBezier(G, H, I, p)
            d4 = sdBezier(I, J, A, p)
            Z[i, j] = min(d1, d2, d3, d4)

    # Compute the gradient of the distance field
    gradient = np.gradient(Z)

    # Plot the gradient field
    plt.figure(figsize=(12, 12))
    plt.contour(X, Y, Z, levels=100, cmap='RdGy')
    plot_curves(A, B, C, F, G, H, I, J)
    plt.streamplot(X, Y, -gradient[1], -gradient[0], linewidth=1, density=2)
    plot_points(A, B, C, F, G, H, I, J)
    plt.title('Gradient Field')
    plt.legend()
    # plt.savefig('sdf_spline_02.png', dpi=300, bbox_inches='tight')
    plt.show()


def poly_spline():
    A = np.array([-0.4, -0.6])  #
    B = np.array([-1.4, 0.0])
    C = np.array([-0.6, 0.4])  #
    F = np.array([0.5, 0.5])
    G = np.array([0.6, 0.75])  #
    H = np.array([0.5, -0.5])
    I = np.array([1.0, -0.5])  #
    J = np.array([0.5, -0.65])
    # contour_poly_spline(A, B, C, F, G, H, I, J)
    plot_gradient_field(A, B, C, F, G, H, I, J)

    # fig, ax = plt.subplots()
    # plot_points(A, B, C, F, G, H)
    # plot_curves(A, B, C, F, G, H)
    # plt.legend()
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    # plt.show()


if __name__ == "__main__":
    poly_spline()
