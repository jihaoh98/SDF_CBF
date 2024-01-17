import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.linalg import block_diag, kron
import pickle
import math


# Binomial function
def binomial(n, i):
    if n >= 0 and i >= 0:
        return math.factorial(n) / (math.factorial(i) * math.factorial(n - i))
    else:
        return 0


# Compute Psi grid
def computePsiGridFast(t, param):
    T0 = np.zeros((len(t), param['nbFct']))
    for n in range(param['nbFct']):
        T0[:, n] = np.power(t, n)
    T = np.kron(np.eye(param['nbSeg']), T0)
    Psi = np.dot(kron(T, T), param['M'])
    return Psi


def computeDistList(Tmat, param):
    if len(Tmat.shape) != 2:
        raise ValueError("Tmat must be a 2D array.")

    T = np.zeros((Tmat.shape[0], param['nbFct']))
    dT = np.zeros_like(T)
    dist = np.zeros(Tmat.shape[1])
    grad = np.zeros((Tmat.shape[1], Tmat.shape[0]))

    for k in range(Tmat.shape[1]):
        id_d = np.zeros(Tmat.shape[0], dtype=int)
        for d in range(Tmat.shape[0]):
            tt = Tmat[d, k] % (1 / param['nbSeg']) * param['nbSeg']
            id_d[d] = round(Tmat[d, k] * param['nbSeg'] - tt)

            if id_d[d] < 0:  # id_id is between [0, nbSeg-1]
                tt += id_d[d]
                id_d[d] = 0

            if id_d[d] > (param['nbSeg'] - 1):
                tt += id_d[d] - (param['nbSeg'] - 1)
                id_d[d] = param['nbSeg'] - 1

            T[d, :] = np.power(tt, np.arange(param['nbFct']))
            dT[d, 1:] = np.arange(1, param['nbFct']) * np.power(tt, np.arange(param['nbFct'] - 1)) * param['nbSeg']

        idtmp1 = id_d[0] * param['nbFct'] + np.arange(param['nbFct'])
        idtmp2 = id_d[1] * param['nbFct'] + np.arange(1, param['nbFct'] + 1)
        idtmp1, idtmp2 = np.meshgrid(idtmp1, idtmp2)

        idtmp = (idtmp1 * param['nbFct'] * param['nbSeg'] + idtmp2 - 1).flatten(
            'F')  # The 'F' option flattens it in column-major like MATLAB

        dist[k] = dist[k] = np.dot(np.kron(T[0, :], T[1, :]).reshape(-1, 1).T, param['Mw'][idtmp]).squeeze()
        grad[k, :] = np.dot(np.vstack((np.kron(dT[0, :], T[1, :]), np.kron(T[0, :], dT[1, :]))),
                            param['Mw'][idtmp]).flatten()

    return dist, grad


def sdf(x):  # generate a shape composed of a circle and a box
    # assume that the robot is a rectangle with length 2m  and width 1m
    length = 0.2
    width = 0.1
    Mu2 = np.array([0.0, 0.0])  # for box

    y = np.zeros((1, x.shape[1]))
    for t in range(x.shape[1]):
        "box sdf"
        dtmp = np.abs(x[:, t] - Mu2) - np.array([width, length])
        d2 = np.linalg.norm(np.maximum(dtmp, 0)) + np.minimum(np.maximum(dtmp[0], dtmp[1]), 0)
        k = 0.1
        h = np.maximum(k - d2, 0)
        y[0, t] = d2 - h ** 2 * 0.25 / k

    return y


def data_generate():
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
        dx[i, :] = (ytmp - y) / e0  # numerical gradients

    # save the sdf data using pickle in data folder
    # save accuracy, domain and y
    with open('data/rectangle_sdf.pkl', 'wb') as f:
        pickle.dump((accuracy, domain, y), f)

    # sdf data visualization
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    ax = fig.add_subplot(111, aspect='equal')
    contours = ax.contour(X1, X2, y.reshape(accuracy, accuracy).T, levels=25, linewidths=2)  # the distance field
    ax.contour(X1, X2, y.reshape(accuracy, accuracy).T, levels=[0], linewidths=2,
               colors='r')  # the shape of the robot
    plt.clabel(contours, inline=True, fontsize=12)
    ax.set_xlim([-domain, domain])
    ax.set_ylim([-domain, domain])
    red_patch = mpatches.Patch(color='red', label='Robot', fill=None, linewidth=2)  # add legend manually
    plt.legend(handles=[red_patch])
    plt.show()


def learning_sdf():
    # load the sdf data
    with open('data/rectangle_sdf.pkl', 'rb') as f:
        accuracy, domain, y = pickle.load(f)
        x_sdf = y.T
    # initialize parameters
    param = {}
    param['nbFct'] = 14  # degree of the Bézier curve
    param['nbSeg'] = 4  # number of segments
    param['nbIn'] = 2  # dimension of the input
    param['nbOut'] = 1  # dimension of the output, sdf is a scalar
    param['nbDim'] = accuracy  # dimension of the grid

    # Pre-computation of Bézier curve in matrix form
    param['B0'] = np.zeros((param['nbFct'], param['nbFct']))
    for n in range(1, param['nbFct'] + 1):
        for i in range(1, param['nbFct'] + 1):
            param['B0'][param['nbFct'] - i, n - 1] = \
                ((-1) ** (param['nbFct'] - i - n + 1)) * \
                binomial(param['nbFct'] - 1, i - 1) * \
                binomial((param['nbFct'] - 1) - (i - 1), (param['nbFct'] - 1) - (n - 1) - (i - 1))

    param['B'] = np.kron(np.eye(param['nbSeg']), param['B0'])
    param['C0'] = block_diag(np.eye(param['nbFct'] - 4), np.array([[1, 0, 0, -1], [0, 1, 1, 2]]).T)
    param['C'] = np.eye(2)
    for n in range(1, param['nbSeg']):
        param['C'] = block_diag(param['C'], param['C0'])
    param['C'] = block_diag(param['C'], np.eye(param['nbFct'] - 2))
    param['M'] = np.kron(np.dot(param['B'], param['C']), np.dot(param['B'], param['C']))

    # Very fast grid computation (using the previously defined computePsiGridFast function)
    nbT = param['nbDim'] / param['nbSeg']
    t = np.linspace(0, 1 - 1 / nbT, int(nbT))
    Psi = computePsiGridFast(t, param)

    # Batch estimation of superposition weights from reference surface
    param['w_sdf'] = np.linalg.lstsq(Psi, x_sdf, rcond=None)[0]

    # Re-encode SDF
    x_sdf = np.dot(Psi, param['w_sdf'])

    # Transformation matrix
    param['Mw'] = np.dot(param['M'], param['w_sdf'])

    # visualize the learned x_sdf
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    ax = fig.add_subplot(111, aspect='equal')
    X1, X2 = np.meshgrid(np.linspace(0, domain - 1 / param['nbDim'], param['nbDim']),
                         np.linspace(0, domain - 1 / param['nbDim'], param['nbDim']))

    contours = ax.contour(X1, X2, x_sdf.reshape(param['nbDim'], param['nbDim']).T, levels=20,
                          linewidths=2)  # the distance field
    ax.contour(X1, X2, x_sdf.reshape(param['nbDim'], param['nbDim']).T, levels=[0], linewidths=2,
               colors='r')  # the shape of the robot
    plt.clabel(contours, inline=True, fontsize=12)
    # ax.set_xlim([0, domain])
    # ax.set_ylim([0, domain])
    plt.show()

    # test the robot x_sdf by querying a point, return distance and gradient
    q = np.array([0.7, 0.3]).reshape(2, 1)
    d_learn, g_learn = computeDistList(q, param)
    print('learned distance: ', d_learn)
    print('learned gradient: ', g_learn)
    # visualize the learned sdf and the gradient of the queried point
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    ax = fig.add_subplot(111, aspect='equal')
    contours = ax.contour(X1, X2, x_sdf.reshape(param['nbDim'], param['nbDim']).T, levels=20,
                          linewidths=2)  # the distance field
    ax.contour(X1, X2, x_sdf.reshape(param['nbDim'], param['nbDim']).T, levels=[0], linewidths=2,
               colors='r')  # the shape of the robot
    plt.clabel(contours, inline=True, fontsize=12)
    ax.set_xlim([0, domain])
    ax.set_ylim([0, domain])
    plt.scatter(q[0], q[1], color='k')
    plt.quiver(q[0], q[1], g_learn[0][0], g_learn[0][1], color='b', scale=10)
    plt.show()


if __name__ == '__main__':
    # data_generate()
    learning_sdf()
