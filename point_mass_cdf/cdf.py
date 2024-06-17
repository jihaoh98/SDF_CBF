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
import matplotlib.cm as cm

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
        self.obj_lists = [Circle(center=torch.tensor([2.0, 1.0]), radius=0.3, device=device)]

        # # # two obstacles case
        # self.obj_lists = [Circle(center=torch.tensor([2.3, -2.3]), radius=0.3, device=device),
        #                   Circle(center=torch.tensor([0.0, 2.4]), radius=0.3, device=device, label='obstacle')]

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

    def inference_t_space_sdf_using_data(self, q):
        # q : (N,2)
        obj_points = torch.cat([obj.sample_surface(70) for obj in self.obj_lists])
        grid = self.x_to_grid(obj_points)
        q_list = (self.q_template[grid[:, 0], grid[:, 1]]).reshape(-1, 2)
        q_list = q_list[q_list[:, 0] != torch.inf]  # filter out the invalid data
        dist = torch.norm(q.unsqueeze(1) - q_list.unsqueeze(0), dim=-1)
        d, min_ind = torch.min(dist, dim=-1)
        q_obj = q_list[min_ind]
        q_obj.requires_grad_(True)
        d_obj = torch.norm(q - q_obj, dim=-1)
        grad_obj = torch.autograd.grad(d_obj, q_obj, torch.ones_like(d), retain_graph=True)[0]
        return d_obj, grad_obj, q_obj

    def inference_c_space_sdf_using_data(self, q, sample_size=70):
        # q : (N,2)
        q.requires_grad = True
        obj_points = torch.cat([obj.sample_surface(sample_size) for obj in self.obj_lists])
        grid = self.x_to_grid(obj_points)
        q_list = (self.q_template[grid[:, 0], grid[:, 1]]).reshape(-1, 2)
        q_list = q_list[q_list[:, 0] != torch.inf]  # filter out the invalid data
        dist = torch.norm(q.unsqueeze(1) - q_list.unsqueeze(0), dim=-1)
        d, min_ind = torch.min(dist, dim=-1)
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

    def plot_cdf(self, d, g, color='k'):
        sdf, grad = self.inference_sdf_grad(self.Q_sets.requires_grad_(True))
        sdf = sdf.detach().cpu().numpy()
        cmap = plt.cm.get_cmap('coolwarm')
        ct = plt.contourf(self.q0, self.q1, d.reshape(self.nbData, self.nbData),
                          cmap=cmap, levels=[-0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], linewidths=1)
        plt.clabel(ct, [], inline=False, fontsize=10, colors='black')
        ct_zero = plt.contour(self.q0, self.q1, sdf.reshape(self.nbData, self.nbData), levels=[0], linewidths=2,
                              colors=color,
                              alpha=1.0)
        for c in ct_zero.collections:
            c.set_hatch('///')  # Apply the hatch patter

        plt.xlabel('$q_0$', size=15)
        plt.ylabel('$q_1$', size=15)

    def plot_non_zero_cdf(self, d, g):
        sdf, grad = self.inference_sdf_grad(self.Q_sets.requires_grad_(True))
        sdf = sdf.detach().cpu().numpy()
        cmap = plt.cm.get_cmap('coolwarm')
        ct = plt.contourf(self.q0, self.q1, d.reshape(self.nbData, self.nbData),
                          cmap=cmap, levels=[-0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], linewidths=1)
        plt.clabel(ct, [], inline=False, fontsize=10, colors='black')
        # ct_zero = plt.contour(self.q0, self.q1, sdf.reshape(self.nbData, self.nbData), levels=[0], linewidths=2,
        #                       colors=color,
        #                       alpha=1.0)
        # for c in ct_zero.collections:
        #     c.set_hatch('///')  # Apply the hatch patter

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

        hatch_handle = mpatches.Patch(facecolor='none', edgecolor='k', hatch='///', label='Obstacle')

        ax.set_xlabel('$q_0$', size=15)
        ax.set_ylabel('$q_1$', size=15)

        return contour, contourf, ct_zero, hatch_handle  # Ensure these match the names used in the animation function

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


def plot_2d_manipulators(link1_length=2, link2_length=2, joint_angles_batch=None):
    # Check if joint_angles_batch is None or has incorrect shape
    if joint_angles_batch is None or joint_angles_batch.shape[1] != 2:
        raise ValueError("joint_angles_batch must be provided with shape (N, 2)")

    # Number of sets of joint angles
    num_sets = joint_angles_batch.shape[0]

    # Create a figure
    cmap = cm.get_cmap('Greens', num_sets)  # You can choose other colormaps like 'Greens', 'Reds', etc.
    cmap2 = cm.get_cmap('Reds', num_sets)  # You can choose other colormaps like 'Greens', 'Reds', etc.
    # the color will
    for i in range(num_sets):
        # Extract joint angles for the current set
        theta1, theta2 = joint_angles_batch[i]

        # Calculate the position of the first joint
        joint1_x = link1_length * np.cos(theta1)
        joint1_y = link1_length * np.sin(theta1)

        # Calculate the position of the end effector (tip of the second link)
        end_effector_x = joint1_x + link2_length * np.cos(theta1 + theta2)
        end_effector_y = joint1_y + link2_length * np.sin(theta1 + theta2)

        # Stack the base, joint, and end effector positions
        positions = np.vstack([[0, 0], [joint1_x, joint1_y], [end_effector_x, end_effector_y]])  # shape: (3, 2)

        # Plotting
        plt.plot(positions[:, 0], positions[:, 1], linestyle='-', color='green', marker='o', markersize=5,
                 markerfacecolor='white',
                 markeredgecolor='green', alpha=0.3)

        # cover the end effector with different colors to hightlight the trajectory
        plt.plot(positions[2, 0], positions[2, 1], linestyle='-', color=cmap(i), marker='o', markersize=5,
                 markerfacecolor='white',
                 markeredgecolor=cmap2(i))
        # plot a bigger base center at (0, 0), which is a cirlce with golden color
        plt.plot(0, 0, marker='o', markersize=15, markerfacecolor='#DDA15E', markeredgecolor='k')


if __name__ == "__main__":
    torch.cuda.empty_cache()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cdf = CDF2D(device)

    # observe the generated data
    cdf.q_template = torch.load(os.path.join(CUR_PATH, 'data2D_100.pt'))
    test_Q1 = torch.tensor([[2., 2.]]).to(device)
    d, grad = cdf.inference_c_space_sdf_using_data(cdf.Q_sets)

    d_test, grad_obj, point_obj = cdf.inference_t_space_sdf_using_data(test_Q1)
    print("distance: ", d_test)
    print("gradient: ", grad_obj)

    cdf.plot_cdf(d.detach().cpu().numpy(), grad.detach().cpu().numpy())
    # plot the test point
    plt.scatter(test_Q1[:, 0].detach().cpu().numpy(), test_Q1[:, 1].detach().cpu().numpy(), color='red')
    # plot the point_obj
    plt.scatter(point_obj[:, 0].detach().cpu().numpy(), point_obj[:, 1].detach().cpu().numpy(), color='red')
    # use fancyarrow to plot the gradient
    arrow = mpatches.FancyArrow(point_obj[0, 0].detach().cpu().numpy(),
                                point_obj[0, 1].detach().cpu().numpy(),
                                grad_obj[0, 0].detach().cpu().numpy(),
                                grad_obj[0, 1].detach().cpu().numpy(),
                                width=0.05,
                                color='red')
    ax = plt.gca()
    ax.add_patch(arrow)

    plt.show()

    plt.figure()
    plt.rcParams['axes.facecolor'] = '#eaeaf2'
    ax = plt.gca()
    for obj in cdf.obj_lists:
        ax.add_patch(obj.create_patch())
    # use fancyarrow to plot the gradient
    # arrow = mpatches.FancyArrow(point_test[0, 0].detach().cpu().numpy(),
    #                             point_test[0, 1].detach().cpu().numpy(),
    #                             grad_test[0, 0].detach().cpu().numpy(),
    #                             grad_test[0, 1].detach().cpu().numpy(),
    #                             width=0.05,
    #                             color='red')
    # ax.add_patch(arrow)
    xf_2d = test_Q1.detach().cpu().numpy()
    xg_2d = point_obj.detach().cpu().numpy()
    plot_2d_manipulators(joint_angles_batch=xf_2d)
    plot_2d_manipulators(joint_angles_batch=xg_2d)

    f_rob_end = cdf.robot.forward_kinematics_all_joints(torch.from_numpy(xf_2d).to(device))[
        0].detach().cpu().numpy()
    plt.scatter(f_rob_end[0, -1], f_rob_end[1, -1], color='r', s=100, zorder=10, label='Goal')
    ax.set_aspect('equal')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend(loc='upper left')
    plt.show()

    # print("distance: ", d_test)
    # print("gradient: ", grad_test)
    # test the norm of the gradient
    # print(torch.norm(grad, dim=-1).max())
    # print(torch.norm(grad, dim=-1).min())
    # print(torch.norm(grad, dim=-1).mean())
    # cdf.plot_sdf()

    # cdf.plot_cdf(d.detach().cpu().numpy(), grad.detach().cpu().numpy())
    # # plot the test point
    # plt.scatter(sample_joint_angles[:, 0].detach().cpu().numpy(), sample_joint_angles[:, 1].detach().cpu().numpy(), color='red')
    # # use the gradient to plot the matches fancy arrow
    # arrow = mpatches.FancyArrow(sample_joint_angles[0, 0].detach().cpu().numpy(),
    #                     sample_joint_angles[0, 1].detach().cpu().numpy(),
    #                     grad_test[0, 0].detach().cpu().numpy(),
    #                     grad_test[0, 1].detach().cpu().numpy(),
    #                     width = 0.05,
    #                     color='red')
    # plt.gca().add_patch(arrow)
    # plt.show()
