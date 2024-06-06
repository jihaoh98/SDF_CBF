import numpy as np
import os
import sys
import torch
import matplotlib.pyplot as plt
from robot2D_torch import Robot2D
from primitives2D_torch import Circle
import math
import time
import matplotlib.patches as mpatches

plt.rcParams['figure.dpi'] = 100
PI = math.pi
CUR_PATH = os.path.dirname(os.path.realpath(__file__))


class CDF2D:
    def __init__(self, device) -> None:
        # super().__init__()
        self.device = device
        self.num_joints = 2
        self.nbData = 100
        self.Q_sets = self.create_Qsets(self.nbData).to(device)
        self.link_length = torch.tensor([[2, 2]]).float().to(device)
        self.obj_center_list = None

        # self.obj_lists = []
        # one obstacle case
        # self.obj_lists = [Circle(center=torch.tensor([2.0, -2.3]), radius=0.3, device=device)]

        # # # two obstacles case
        self.obj_lists = [Circle(center=torch.tensor([2.3, -2.3]), radius=0.3, device=device),
                          Circle(center=torch.tensor([0.0, 2.4]), radius=0.3, device=device, label='obstacle')]

        # three obstacles case
        # self.obj_lists = [Circle(center=torch.tensor([2.3, -2.3]), radius=0.3, device=device),
        #                     Circle(center=torch.tensor([0.0, 2.45]), radius=0.3, device=device),
        #                     Circle(center=torch.tensor([2.5, 1.5]), radius=0.3, device=device, label='obstacle'),]

        self.q_max = torch.tensor([PI, PI]).to(device)
        self.q_min = torch.tensor([-PI, -PI]).to(device)

        # data generation
        self.workspace = [[-3.0, -3.0], [3.0, 3.0]]
        self.batchsize = 20000  # batch size of q
        self.epsilon = 1e-3  # distance threshold to filter data

        # robot
        self.robot = Robot2D(num_joints=self.num_joints, init_states=self.Q_sets, link_length=self.link_length,
                             device=device)

        # # c space distance field
        self.q_template = torch.load(os.path.join(CUR_PATH, 'data2D_100.pt'))
        # self.q_template = None

    def add_object(self, obj):
        self.obj_lists.append(obj)

    def create_grid(self, nb_data):
        t = np.linspace(-math.pi, math.pi, nb_data)
        self.q0, self.q1 = np.meshgrid(t, t)
        return self.q0, self.q1

    def create_Qsets(self, nb_data):
        q0, q1 = self.create_grid(nb_data)
        q0_torch = torch.from_numpy(q0).float()
        q1_torch = torch.from_numpy(q1).float()
        Q_sets = torch.cat([q0_torch.unsqueeze(-1), q1_torch.unsqueeze(-1)], dim=-1).view(-1, 2)
        return Q_sets

    def inference_sdf(self, q):
        # using predefined object
        kpts = self.robot.surface_points_sampler(q)
        B, N = kpts.size(0), kpts.size(1)
        dist = torch.cat([obj.signed_distance(kpts.reshape(-1, 2)).reshape(B, N, -1) for obj in self.obj_lists], dim=-1)
        # using the closest point from robot surface
        sdf = torch.min(dist, dim=-1)[0]
        sdf = sdf.min(dim=-1)[0]
        return sdf

    def inference_sdf_grad(self, q):
        # using predefined object
        q.requires_grad = True
        kpts = self.robot.surface_points_sampler(q)
        B, N = kpts.size(0), kpts.size(1)
        dist = torch.cat([obj.signed_distance(kpts.reshape(-1, 2)).reshape(B, N, -1) for obj in self.obj_lists], dim=-1)
        # using the closest point from robot surface
        sdf = torch.min(dist, dim=-1)[0]
        sdf = sdf.min(dim=-1)[0]
        grad = torch.autograd.grad(sdf, q, torch.ones_like(sdf), create_graph=True)[0]
        return sdf, grad

    def c_space_distance(self, q):
        # x : (Nx,3)
        # q : (Np,7)
        # return d : (Np) distance between q and x in C space. d = min_{q*}{L2(q-q*)}. sdf(x,q*)=0

        # compute d
        Np = q.shape[0]
        dist = torch.norm(q.unsqueeze(1) - self.q_template.unsqueeze(0), dim=-1)
        d = torch.min(dist, dim=-1)[0]

        # compute sign of d
        d_ts = self.inference_sdf(q)
        mask = (d_ts < 0)
        d[mask] = -d[mask]
        return d

    def x_to_grid(self, p):
        # p: (N,2)
        # return grid index (N,2)
        x_workspace = torch.tensor([self.workspace[0][0], self.workspace[1][0]]).to(self.device)
        y_workspace = torch.tensor([self.workspace[0][1], self.workspace[1][1]]).to(self.device)

        x_grid = (p[:, 0] - x_workspace[0]) / (x_workspace[1] - x_workspace[0]) * self.nbData
        y_grid = (p[:, 1] - y_workspace[0]) / (y_workspace[1] - y_workspace[0]) * self.nbData

        x_grid.clamp_(0, self.nbData - 1)
        y_grid.clamp_(0, self.nbData - 1)
        return torch.stack([x_grid, y_grid], dim=-1).long()

    def inference_c_space_sdf_using_data(self, q):
        # q : (N,2)
        q.requires_grad = True
        obj_points = torch.cat([obj.sample_surface(70) for obj in self.obj_lists])
        grid = self.x_to_grid(obj_points)
        q_list = (self.q_template[grid[:, 0], grid[:, 1]]).reshape(-1, 2)
        q_list = q_list[q_list[:, 0] != torch.inf]  # filter out the invalid data
        dist = torch.norm(q.unsqueeze(1) - q_list.unsqueeze(0), dim=-1)
        d = torch.min(dist, dim=-1)[0]
        grad = torch.autograd.grad(d, q, torch.ones_like(d), retain_graph=True)[0]
        return d, grad

    def plot_sdf(self):
        sdf, grad = self.inference_sdf_grad(self.Q_sets.requires_grad_(True))
        sdf = sdf.detach().cpu().numpy()
        cmap = plt.cm.get_cmap('coolwarm')
        ct = plt.contourf(self.q0, self.q1, sdf.reshape(self.nbData, self.nbData),
                          cmap=cmap, levels=[-0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], linewidths=1)
        plt.clabel(ct, [], inline=False, fontsize=10, colors='black')
        ct_zero = plt.contour(self.q0, self.q1, sdf.reshape(self.nbData, self.nbData), levels=[0], linewidths=2,
                              colors='k',
                              alpha=1.0)
        for c in ct_zero.collections:
            c.set_hatch('///')  # Apply the hatch pattern
        plt.xlabel('$q_0$', size=15)
        plt.ylabel('$q_1$', size=15)

    def plot_cdf(self, d, g):
        sdf, grad = self.inference_sdf_grad(self.Q_sets.requires_grad_(True))
        sdf = sdf.detach().cpu().numpy()
        cmap = plt.cm.get_cmap('coolwarm')
        ct = plt.contourf(self.q0, self.q1, d.reshape(self.nbData, self.nbData),
                          cmap=cmap, levels=[-0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], linewidths=1)
        plt.clabel(ct, [], inline=False, fontsize=10, colors='black')
        ct_zero = plt.contour(self.q0, self.q1, sdf.reshape(self.nbData, self.nbData), levels=[0], linewidths=2,
                              colors='k',
                              alpha=1.0)
        for c in ct_zero.collections:
            c.set_hatch('///')  # Apply the hatch patter

        # use streamline to plot gradient field
        # x = np.linspace(-PI, PI, self.nbData)
        # y = np.linspace(-PI, PI, self.nbData)
        # X, Y = np.meshgrid(x, y)
        # U = g[:, 0].reshape(self.nbData, self.nbData)
        # V = g[:, 1].reshape(self.nbData, self.nbData)
        # # Create the streamplot and get the returned LineCollection object
        # streams = plt.streamplot(X, Y, U, V, color='blue', linewidth=0.5)

        plt.xlabel('$q_0$', size=15)
        plt.ylabel('$q_1$', size=15)

    def plot_cdf_ax(self, d, ax):
        sdf, grad = self.inference_sdf_grad(self.Q_sets.requires_grad_(True))
        sdf = sdf.detach().cpu().numpy()

        # Make sure to use the axes object (ax) for plotting
        contour = ax.contour(self.q0, self.q1, sdf.reshape(self.nbData, self.nbData), levels=[0], linewidths=2,
                             colors='k', alpha=1.0)
        cmap = plt.cm.get_cmap('coolwarm')
        contourf = ax.contourf(self.q0, self.q1, d.reshape(self.nbData, self.nbData),
                               cmap=cmap, levels=[-0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 6.0])
        ax.clabel(contourf, [], inline=False, fontsize=10)

        ct_zero = ax.contour(self.q0, self.q1, sdf.reshape(self.nbData, self.nbData), levels=[0], colors='k')
        for c in ct_zero.collections:
            c.set_hatch('///')

        ax.set_xlabel('$q_0$', size=15)
        ax.set_ylabel('$q_1$', size=15)

        return contour, contourf, ct_zero  # Ensure these match the names used in the animation function

    def plot_sdf_ax(self, ax):
        sdf, grad = self.inference_sdf_grad(self.Q_sets.requires_grad_(True))
        sdf = sdf.detach().cpu().numpy()
        cmap = plt.cm.get_cmap('coolwarm')

        # Make sure to use the axes object (ax) for plotting
        contour = ax.contour(self.q0, self.q1, sdf.reshape(self.nbData, self.nbData), levels=[0], linewidths=2,
                             colors='k', alpha=1.0)
        contourf = ax.contourf(self.q0, self.q1, sdf.reshape(self.nbData, self.nbData),
                               cmap=cmap, levels=[-0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 6.0])
        ax.clabel(contourf, [], inline=False, fontsize=10)

        ct_zero = ax.contour(self.q0, self.q1, sdf.reshape(self.nbData, self.nbData), levels=[0], colors='k')
        for c in ct_zero.collections:
            c.set_hatch('///')

        ax.set_xlabel('$q_0$', size=15)
        ax.set_ylabel('$q_1$', size=15)

        return contour, contourf, ct_zero  # Ensure these match the names used in the animation function

    def plot_robot(self, q, color='black'):
        # plot robot
        for _q in q:
            f_rob = self.robot.forward_kinematics_all_joints(_q.unsqueeze(0))[0].detach().cpu().numpy()
            plt.plot(f_rob[0, :], f_rob[1, :], color=color)  # plot link
            plt.plot(f_rob[0, 0], f_rob[1, 0], '.', color='black', markersize=10)  # plot joint

    def plot_objects(self):
        for obj in self.obj_lists:
            plt.gca().add_patch(obj.create_patch())


if __name__ == "__main__":
    torch.cuda.empty_cache()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cdf = CDF2D(device)

    # observe the generated data
    cdf.q_template = torch.load(os.path.join(CUR_PATH, 'data2D_100.pt'))
    d, grad = cdf.inference_c_space_sdf_using_data(cdf.Q_sets)
    # manually test some points on the grid to get the distance and gradient
    test_Q1 = torch.tensor([[-2., -2.]]).to(device)
    d_test, grad_test = cdf.inference_c_space_sdf_using_data(test_Q1)

    print("distance: ", d_test)
    print("gradient: ", grad_test)
    # test the norm of the gradient
    print(torch.norm(grad, dim=-1).max())
    print(torch.norm(grad, dim=-1).min())
    print(torch.norm(grad, dim=-1).mean())
    # cdf.plot_sdf()
    # plt.show()

    cdf.plot_cdf(d.detach().cpu().numpy(), grad.detach().cpu().numpy())
    # # plot the test point
    # plt.scatter(test_Q1[:, 0].detach().cpu().numpy(), test_Q1[:, 1].detach().cpu().numpy(), color='red')
    # # use the gradient to plot the matches fancy arrow
    # arrow = mpatches.FancyArrow(test_Q1[0, 0].detach().cpu().numpy(),
    #                     test_Q1[0, 1].detach().cpu().numpy(),
    #                     grad_test[0, 0].detach().cpu().numpy(),
    #                     grad_test[0, 1].detach().cpu().numpy(),
    #                     width = 0.05,
    #                     color='red')
    # plt.gca().add_patch(arrow)
    plt.show()
