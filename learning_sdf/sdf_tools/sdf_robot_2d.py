import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def demo_sdf01():
    accuracy = 100  # Number of datapoints per axis for visualization
    domain = 1.0
    X1, X2 = np.meshgrid(np.linspace(-domain, domain, accuracy), np.linspace(-domain, domain, accuracy))
    x = np.vstack((X1.T.ravel(), X2.T.ravel()))
    y = sdf(x)  # compute the sdf
    e0 = 1E-6
    dx = np.zeros((x.shape[0], accuracy ** 2))
    for i in range(x.shape[0]):
        e = np.zeros((x.shape[0], 1))
        e[i] = e0
        ytmp = sdf(x + np.tile(e, (1, accuracy ** 2)))
        dx[i, :] = (y - ytmp) / e0  # numerical gradients

    fig = plt.figure(figsize=(12, 8), facecolor='white')
    ax = fig.add_subplot(111, aspect='equal')
    contours = ax.contour(X1, X2, y.reshape(accuracy, accuracy).T, levels=15, linewidths=2)  # the distance field
    ax.contour(X1, X2, y.reshape(accuracy, accuracy).T, levels=[0], linewidths=2,
               colors='r')  # the shape of the robot
    plt.clabel(contours, inline=True, fontsize=12)
    ax.set_xlim([-domain, domain])
    ax.set_ylim([-domain, domain])
    red_patch = mpatches.Patch(color='red', label='Robot', fill=None, linewidth=2)  # add legend manually
    plt.legend(handles=[red_patch])
    plt.show()


def sdf(x):  # generate a shape composed of a circle and a box

    Mu1 = np.array([0.0, 0.0])  # for circle
    Mu2 = np.array([0.3, 0.4])  # for box

    y = np.zeros((1, x.shape[1]))
    for t in range(x.shape[1]):
        "circle"
        radius = 0.3
        d1 = (np.sqrt(np.dot((Mu1 - x[:, t]).T, (Mu1 - x[:, t]))) - radius)

        "box"
        length = radius
        width = radius / 2
        dtmp = np.abs(x[:, t] - Mu2) - np.array([width, length])
        d2 = np.linalg.norm(np.maximum(dtmp, 0)) + np.minimum(np.maximum(dtmp[0], dtmp[1]), 0)
        k = 0.1
        h = np.maximum(k - np.abs(d1 - d2), 0)
        y[0, t] = np.minimum(d1, d2) - h ** 2 * 0.25 / k

    return y


if __name__ == '__main__':
    demo_sdf01()
