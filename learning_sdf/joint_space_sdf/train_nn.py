# 2D example for configuration space distance field

import numpy as np
import os
import sys
import torch
import math
import matplotlib.pyplot as plt
import time
import math

from matplotlib.animation import FuncAnimation

from cdf import CDF2D

PI = math.pi
CUR_PATH = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(os.path.join(CUR_PATH,'../../RDF'))
from mlp import MLPRegression

# from Manifold.examples.stochman.manifold import Manifold

torch.manual_seed(10)


class Trainer:
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device
        # c space distance field
        self.cdf = CDF2D(device)
        # self.obstacle = self.cdf.obj_lists[0].sample_surface(100)
        # sample all the obstacles
        self.obstacle = torch.cat([obj.sample_surface(100) for obj in self.cdf.obj_lists], dim=0)
        # self.q = self.cdf.q_template.view(-1, 200, 2)
        self.Q_sets = self.cdf.Q_sets
        self.nbData = self.cdf.nbData
        self.q0, self.q1 = self.cdf.create_grid(self.nbData)

        x = torch.linspace(self.cdf.workspace[0][0], self.cdf.workspace[1][0], self.cdf.nbData).to(self.device)
        y = torch.linspace(self.cdf.workspace[0][1], self.cdf.workspace[1][1], self.cdf.nbData).to(self.device)
        xx, yy = torch.meshgrid(x, y)
        xx, yy = xx.reshape(-1, 1), yy.reshape(-1, 1)
        self.p = torch.cat([xx, yy], dim=-1).to(self.device)

    def matching_csdf(self, q):
        # q: [batchsize,2]
        # return d:[len(x),len(q)]
        dist = torch.norm(q.unsqueeze(1).expand(-1, 200, -1) - self.q.unsqueeze(1), dim=-1)
        d, idx = torch.min(dist, dim=-1)
        q_template = torch.gather(self.q, 1, idx.unsqueeze(-1).expand(-1, -1, 2))
        return d, q_template

    def train_nn(self, batchsize=100, epoches=500):

        # model
        model = MLPRegression(input_dims=4, output_dims=1, mlp_layers=[256, 256, 128, 128, 128], skips=[],
                              act_fn=torch.nn.ReLU, nerf=True)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5000,
                                                               threshold=0.01, threshold_mode='rel',
                                                               cooldown=0, min_lr=0, eps=1e-04, verbose=True)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        batch_p = self.p.unsqueeze(1).expand(-1, batchsize, -1).reshape(-1, 2)
        model_dict = {}
        for iter in range(epoches):
            model.train()
            with torch.cuda.amp.autocast():
                q = torch.rand(batchsize, 2).to(self.device) * 2 * math.pi - math.pi
                batch_q = q.unsqueeze(0).expand(len(self.p), -1, -1).reshape(-1, 2)
                batch_q.requires_grad = True
                d, q_temp = self.matching_csdf(q)
                q_temp = q_temp.reshape(-1, 2)
                mask = d.reshape(-1) < torch.inf
                # mask = d<torch.inf
                # print(d.shape,_p.shape,q.shape)
                inputs = torch.cat([batch_p, batch_q], dim=-1).reshape(-1, 4)
                outputs = d.reshape(-1, 1)
                inputs, outputs = inputs[mask], outputs[mask]
                q_temp = q_temp[mask]
                # batch_q = batch_q[mask]
                weights = torch.ones_like(outputs).to(device)
                # weights = (1/outputs).clamp(0,1)

                d_pred = model.forward(inputs)
                d_grad_pred = torch.autograd.grad(d_pred, batch_q, torch.ones_like(d_pred), retain_graph=True)[0]
                d_grad_pred = d_grad_pred[mask]

                # Compute the Eikonal loss
                eikonal_loss = torch.abs(d_grad_pred.norm(2, dim=-1) - 1).mean()

                # Compute the MSE loss
                d_loss = ((d_pred - outputs) ** 2 * weights).mean()

                # Compute the projection loss
                proj_q = batch_q[mask] - d_grad_pred * d_pred
                proj_loss = torch.norm(proj_q - q_temp, dim=-1).mean()

                # torch.nn.functional.mse_loss(sdf_pred, outputs, reduction='mean')

                # Combine the two losses with appropriate weights
                w0 = 1.0
                w1 = 1.0
                w2 = 0.1
                loss = w0 * d_loss + w1 * eikonal_loss + w2 * proj_loss

                # # Print the losses for monitoring

                # loss = 0.1*d_loss 
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step(loss)
                if iter % 100 == 0:
                    print(
                        f"Epoch:{iter}\tMSE Loss: {d_loss.item()}\tEikonal Loss: {eikonal_loss.item()}\tProjection Loss: {proj_loss.item()}\tTotal loss:{loss.item()}")
                    model_dict[iter] = model.state_dict()
                    torch.save(model_dict, os.path.join(CUR_PATH, 'model_dict.pt'))
        return model

    def inference(self, q, model):
        model.eval()
        x, q = self.obstacle.to(self.device), q.to(self.device)
        q.requires_grad = True
        x_cat = x.unsqueeze(1).expand(-1, len(q), -1).reshape(-1, 2)
        q_cat = q.unsqueeze(0).expand(len(x), -1, -1).reshape(-1, 2)
        inputs = torch.cat([x_cat, q_cat.type(torch.float32)], dim=-1)
        sdf_pred = model.forward(inputs)  # min operation, min {d1, d2, d3}
        return sdf_pred

    def inference_plot(self, q, model):
        model.eval()
        x, q = self.obstacle.to(self.device), q.to(self.device)
        x_cat = x.unsqueeze(1).expand(-1, len(q), -1).reshape(-1, 2)
        q_cat = q.unsqueeze(0).expand(len(x), -1, -1).reshape(-1, 2)
        inputs = torch.cat([x_cat, q_cat.type(torch.float32)], dim=-1)
        sdf_pred = model.forward(inputs)
        return sdf_pred

    def inference_d_wrt_q_plot(self, q, model):
        sdf_pred = self.inference_plot(q, model)
        # sdf_sign, _ = self.cdf.inference_sdf(q)
        # mask = sdf_sign < 0
        d = sdf_pred.reshape(len(self.obstacle), len(q)).min(dim=0)[0]
        grad = torch.autograd.grad(d, q, torch.ones_like(d), retain_graph=True)[0]
        grad = torch.nn.functional.normalize(grad, dim=-1)
        # d[mask] = -d[mask]
        # grad[mask] = -grad[mask]
        return d, grad

    def inference_d_wrt_q(self, q, model):
        sdf_pred = self.inference(q, model)
        # sdf_sign, _ = self.cdf.inference_sdf(q)
        # mask = sdf_sign < 0
        d = sdf_pred.reshape(len(self.obstacle), len(q)).min(dim=0)[0]
        grad = torch.autograd.grad(d, q, torch.ones_like(d), retain_graph=True)[0]
        # d[mask] = -d[mask]
        # grad[mask] = -grad[mask]
        return d, grad

    def inference_d_wrt_x(self, q, model):
        sdf_pred = self.inference(self.obstacle, q, model)
        d = sdf_pred.reshape(len(self.obstacle), len(q)).min(dim=1)[0]
        grad = torch.autograd.grad(d, self.obstacle, torch.ones_like(d), retain_graph=True)[0]
        return d, grad

    def projection(self, q, d, grad):
        # q : (N,2)
        # d : (N)
        # grad : (N,2)
        # return q_proj : (N,2)
        q_proj = q - grad * d.unsqueeze(-1)
        return q_proj

    def plot_sdf(self):
        sdf = self.inference_d_wrt_q(self.obstacle, self.Q_sets)
        sdf = sdf.detach().cpu().numpy()
        plt.contour(self.q0, self.q1, sdf.reshape(self.nbData, self.nbData), levels=[0], linewidths=4, colors='b',
                    alpha=1.0)
        ct = plt.contour(self.q0, self.q1, sdf.reshape(self.nbData, self.nbData), levels=10, linewidths=1)
        plt.clabel(ct, inline=False, fontsize=10)
        plt.xlabel('q0', size=15)
        plt.ylabel('q1', size=15)

    def plot_c_space_distance(self, model):
        q_proj = self.Q_sets  # 网格划分，拿到采样点
        q_proj.requires_grad = True
        for i in range(5):
            d, grad = self.inference_d_wrt_q_plot(q_proj, model)  # get the distance and gradient
            if i == 0:
                d0 = d
            q_proj = self.projection(q_proj, d, grad)
        plt.plot(q_proj[:, 0].detach().cpu().numpy(), q_proj[:, 1].detach().cpu().numpy(), '.', color='red')
        d0 = d0.detach().cpu().numpy()
        plt.contour(self.q0, self.q1, d0.reshape(self.nbData, self.nbData), levels=[0], linewidths=4, colors='b',
                    alpha=1.0)
        ct = plt.contour(self.q0, self.q1, d0.reshape(self.nbData, self.nbData), levels=10, linewidths=1)
        plt.clabel(ct, inline=False, fontsize=10)
        plt.xlabel('q0', size=15)
        plt.ylabel('q1', size=15)

    def plot_c_space_distance_wrt_x(self, model):
        q = torch.ones(1, 2).to(self.device)
        self.p.requires_grad = True
        d, grad = self.inference_d_wrt_x(self.p, q, model)
        d = d.detach().cpu().numpy()
        plt.contour(self.q0, self.q1, d.reshape(self.nbData, self.nbData), levels=[0], linewidths=4, colors='b',
                    alpha=1.0)
        ct = plt.contour(self.q0, self.q1, d.reshape(self.nbData, self.nbData), levels=10, linewidths=1)
        plt.clabel(ct, inline=False, fontsize=10)
        plt.xlabel('x0', size=15)
        plt.ylabel('x1', size=15)

    def plot_robot(self, q, color='black'):
        # plot robot
        for _q in q:
            f_rob = self.robot.forward_kinematics_all_joints(_q.unsqueeze(0))[0].detach().cpu().numpy()
            plt.plot(f_rob[0, :], f_rob[1, :], color=color)  # plot link
            plt.plot(f_rob[0, 0], f_rob[1, 0], '.', color='black', markersize=10)  # plot joint

    def plot_objects(self):
        for obj in self.obj_lists:
            plt.gca().add_patch(obj.create_patch())

    def animate_iteration(self, geodesics):
        fig, ax = plt.subplots()
        # plot the obstacles in joint space
        sdf, grad = self.inference_d_wrt_q(self.obstacle, self.Q_sets.requires_grad_(True), model)
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
                   grad[:, 1].reshape(self.nbData, self.nbData), color='k')
        line, = ax.plot([], [], lw=2, color="blue")
        tensor_list = [item[0] for item in geodesics.record]
        final_tensor = torch.stack(tensor_list)  # Stacking them together

        def init():
            """Initialize the line and point with empty data."""
            line.set_data([], [])
            return line,

        def update(num):
            """Update the line and point for frame i."""
            line.set_data(final_tensor[num, :, 0].cpu().detach().numpy(),
                          final_tensor[num, :, 1].cpu().detach().numpy())
            # ax.set_ylim(1.1, -0.1)  # Set the limits in reverse order
            return line,

        # ani = FuncAnimation(fig, update, frames=len(curve_geodesics.record), init_func=init, blit=True)
        # ani.save('animation.mp4', writer='ffmpeg', fps=30)
        # ani.save('animation.gif', writer='pillow', fps=int(30))
        # plt.show()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(device)
    # model = trainer.train_nn(epoches=5000)
    # torch.save(model.state_dict(), os.path.join(CUR_PATH,'model.pt'))
    model = MLPRegression(input_dims=4, output_dims=1, mlp_layers=[256, 256, 128, 128, 128], skips=[],
                          act_fn=torch.nn.ReLU, nerf=True)
    model.load_state_dict(torch.load(os.path.join(CUR_PATH, 'model.pt')))
    model.to(device)
    trainer.plot_c_space_distance(model)
    # trainer.plot_c_space_distance_wrt_x(model)
    plt.show()
