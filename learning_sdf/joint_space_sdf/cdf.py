import numpy as np
import os
import sys
import torch
import matplotlib.pyplot as plt
from robot2D_torch import Robot2D
from primitives2D_torch import Box, Circle, Triangle, Ellipse, Union, Erode
import math
import matplotlib.animation as animation
plt.rcParams['figure.dpi'] = 200

PI = math.pi
CUR_PATH = os.path.dirname(os.path.realpath(__file__))


class CDF2D:
    def __init__(self, device) -> None:
        # super().__init__()
        self.device = device
        self.num_joints = 2
        self.nbData = 50
        self.Q_sets = self.create_Qsets(self.nbData).to(device)
        self.link_length = torch.tensor([[2, 2]]).float().to(device)
        self.obj_center_list = None
        self.obj_lists = [Circle(center=torch.tensor([2.0, -2.35]), radius=0.3, device=device)]
        # self.obj_lists = [Circle(center=torch.tensor([2.3, 1.35]), radius=0.3, device=device),
        #                   Circle(center=torch.tensor([0.0, -2.35]), radius=0.3, device=device)]
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
        # self.q_template = torch.load(os.path.join(CUR_PATH, 'data2D.pt'))

    def metric(self, c=None):
        M = []
        for _, xi in enumerate(c):
            d, g = self.inference_sdf(xi.reshape(1, 2))

            scMin = 0.01
            sc = 1 / (1 + torch.exp(-1e2 * (d - 0.2))) * (1 - scMin) + scMin
            R = torch.column_stack((torch.tensor([g[0][0], g[0][1]]), torch.tensor([-g[0][1], g[0][0]]))).to(
                self.device)
            m = torch.ger(R[:, 0], R[:, 0]) * sc + torch.ger(R[:, 1], R[:, 1])
            if d < 0.0:
                m = torch.linalg.inv(torch.eye(2).to(self.device) * 1e1)  # homogenous metric
            else:
                m = torch.linalg.pinv(m)
            M.append(m)
        return torch.stack(M, dim=0)

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

        obj_points = torch.cat([obj.sample_surface(200) for obj in self.obj_lists])
        grid = self.x_to_grid(obj_points)

        q_list = (self.q_template[grid[:, 0], grid[:, 1]]).reshape(-1, 2)
        q_list = q_list[q_list[:, 0] != torch.inf]
        dist = torch.norm(q.unsqueeze(1) - q_list.unsqueeze(0), dim=-1)
        d = torch.min(dist, dim=-1)[0]
        return d, q_list

    def plot_sdf(self):
        sdf, grad = self.inference_sdf(self.Q_sets.requires_grad_(True))
        sdf = sdf.detach().cpu().numpy()
        plt.contour(self.q0, self.q1, sdf.reshape(self.nbData, self.nbData), levels=[0], linewidths=4, colors='b',
                    alpha=1.0)
        ct = plt.contour(self.q0, self.q1, sdf.reshape(self.nbData, self.nbData), levels=10, linewidths=1)
        plt.clabel(ct, inline=False, fontsize=10)
        plt.xlabel('q0', size=15)
        plt.ylabel('q1', size=15)

        # plot gradient
        grad = grad.detach().cpu().numpy()
        plt.quiver(self.q0, self.q1, grad[:, 0].reshape(self.nbData, self.nbData),
                   grad[:, 1].reshape(self.nbData, self.nbData), color='r')

    def plot_c_space_distance(self, d):
        plt.contour(self.q0, self.q1, d.reshape(self.nbData, self.nbData), levels=[0], linewidths=4, colors='b',
                    alpha=1.0)
        ct = plt.contour(self.q0, self.q1, d.reshape(self.nbData, self.nbData), levels=10, linewidths=1)
        plt.clabel(ct, inline=False, fontsize=10)
        plt.xlabel('q0', size=15)
        plt.ylabel('q1', size=15)

    def plot_robot(self, q, color='black'):
        # plot robot
        for _q in q:
            f_rob = self.robot.forward_kinematics_all_joints(_q.unsqueeze(0))[0].detach().cpu().numpy()
            plt.plot(f_rob[0, :], f_rob[1, :], color=color)  # plot link
            plt.plot(f_rob[0, 0], f_rob[1, 0], '.', color='black', markersize=10)  # plot joint

    def plot_objects(self):
        for obj in self.obj_lists:
            plt.gca().add_patch(obj.create_patch())

    def plot_robot_trajectory(self, points, ax):
        # plot obstacles

        for obj in self.obj_lists:
            ax.add_patch(obj.create_patch())
        # plot robot
        k = 10  # Replace 5 with whatever step size you want
        for i in range(0, len(points), k):
            _q = points[i]
            f_rob = self.robot.forward_kinematics_all_joints(_q.unsqueeze(0))[0].detach().cpu().numpy()
            plt.plot(f_rob[0, :], f_rob[1, :], color='black')
            plt.pause(0.001)

    def anima_robot_trajectory(self, geodesic):
        # Animation setup
        t = torch.linspace(0.0, 1.0, 100).to(self.device)
        curve_points = geodesic(t)
        fig, ax = plt.subplots()

        # Initialize trajectory line
        line, = ax.plot([], [], color='black')

        # Function to update the line in each frame
        def update(frame):
            _q = curve_points[frame]
            f_rob = self.robot.forward_kinematics_all_joints(_q.unsqueeze(0))[0].detach().cpu().numpy()
            line.set_data(f_rob[0, :], f_rob[1, :])
            return line,

        # # Plotting obstacles
        # for obj in self.obj_lists:
        #     ax.add_patch(obj.create_patch())

        ani = animation.FuncAnimation(fig, update, frames=len(curve_points), blit=True)

        # Show or save animation
        plt.show()
        # ani.save('animation.mp4')  # Uncomment to save the animation


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cdf = CDF2D(device)
    q = torch.tensor([0.0, torch.deg2rad(torch.tensor(45))]).view(1, 2).to(device)
    cdf.plot_sdf()
    # d = cdf.c_space_distance(q)
    # cdf.plot_c_space_distance(d.detach().cpu().numpy())
    # cdf.plot_objects()
    # cdf.plot_robot(q)
    plt.axis('equal')
    plt.show()
