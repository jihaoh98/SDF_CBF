# 7D panda robot

import numpy as np
import os
import sys
import torch
import math
import matplotlib.pyplot as plt
import time
import math
import trimesh

from mlp import MLPRegression

CUR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CUR_PATH, 'RDF'))
import bf_sdf
from RDF.panda_layer.panda_layer import PandaLayer

sys.path.append(os.path.join(CUR_PATH, 'Neural-JSDF/learning/nn-learning'))
from sdf.robot_sdf import RobotSdfCollisionNet

# import bf_sdf

sys.path.append(os.path.join(CUR_PATH, 'Neural-JSDF/learning/nn-learning'))
# from sdf.robot_sdf import RobotSdfCollisionNet
# from torchmin import minimize

PI = math.pi
# torch.manual_seed(10)
np.random.seed(10)


# torch.autograd.set_detect_anomaly(True)

class CDF:
    def __init__(self, device) -> None:
        # device
        self.device = device

        # data
        # self.raw_data = np.load(os.path.join(CUR_PATH,'data.npy'),allow_pickle=True).item()
        self.batch_x = 10
        self.batch_q = 100
        self.max_q_per_link = 100

        # self.process_data(self.raw_data)
        # self.data = self.load_data()
        # self.len_data = len(self.data['k'])

        # self.np_csdf_data = self.load_np_csdf_data()
        # panda robot
        self.panda = PandaLayer(device, mesh_path=os.path.join(CUR_PATH, 'RDF/panda_layer/meshes/smooth/*.stl'))

        self.bp_sdf = bf_sdf.BPSDF(8, -1.0, 1.0, self.panda, device)
        self.bp_sdf_model = torch.load(os.path.join(CUR_PATH, 'RDF/models/BP_8.pt'))

        tensor_args = {'device': device, 'dtype': torch.float32}
        self.njsdf_model = RobotSdfCollisionNet(in_channels=10, out_channels=9, layers=[256] * 4, skips=[])
        self.njsdf_model.load_weights(os.path.join(CUR_PATH, 'Neural-JSDF/learning/nn-learning/sdf_256x5_mesh.pt'),
                                      tensor_args)
        self.njsdf_model = self.njsdf_model.model

    def process_data(self, data):
        keys = list(data.keys())  # Create a copy of the keys
        processed_data = {}
        for k in keys:
            if len(data[k]['q']) == 0:
                data.pop(k)
                continue
            q = torch.from_numpy(data[k]['q']).float().to(self.device)
            q_idx = torch.from_numpy(data[k]['idx']).float().to(self.device)
            q_idx[q_idx == 7] = 6
            q_idx[q_idx == 8] = 7
            q_lib = torch.inf * torch.ones(self.max_q_per_link, 7, 7).to(self.device)
            for i in range(1, 8):
                mask = (q_idx == i)
                if len(q[mask]) > self.max_q_per_link:
                    fps_q = pytorch3d.ops.sample_farthest_points(q[mask].unsqueeze(0), K=self.max_q_per_link)[0]
                    q_lib[:, :, i - 1] = fps_q.squeeze()
                    # print(q_lib[:,:,i]) 
                elif len(q[mask]) > 0:
                    q_lib[:len(q[mask]), :, i - 1] = q[mask]

            processed_data[k] = {
                'x': torch.from_numpy(data[k]['x']).float().to(self.device),
                'q': q_lib,
            }
        final_data = {
            'x': torch.cat([processed_data[k]['x'].unsqueeze(0) for k in processed_data.keys()], dim=0),
            'q': torch.cat([processed_data[k]['q'].unsqueeze(0) for k in processed_data.keys()], dim=0),
            'k': torch.tensor([k for k in processed_data.keys()]).to(self.device)
        }

        torch.save(final_data, os.path.join(CUR_PATH, 'data.pt'))
        return data

    def load_data(self):
        data = torch.load(os.path.join(CUR_PATH, 'data.pt'))
        return data

    def select_data(self):
        # x_batch:(batch_x,3)
        # q_batch:(batch_q,7)
        # d:(batch_x,batch_q)

        x = self.data['x']
        q = self.data['q']

        idx = torch.randint(0, len(x), (self.batch_x,))
        # idx = torch.tensor([4000])
        x_batch, q_lib = x[idx], q[idx]
        # print(x_batch)
        q_batch = self.sample_q()
        d, grad = self.decode_distance(x_batch, q_batch, q_lib)
        return x_batch, q_batch, d, grad

    def decode_distance(self, q_batch, q_lib):
        # batch_q:(batch_q,7)
        # q_lib:(batch_x,self.max_q_per_link,7,7)

        batch_x = q_lib.shape[0]
        batch_q = q_batch.shape[0]
        d_tensor = torch.ones(batch_x, batch_q, 7).to(self.device) * torch.inf
        grad_tensor = torch.zeros(batch_x, batch_q, 7, 7).to(self.device)
        for i in range(7):
            q_lib_temp = q_lib[:, :, :i + 1, i].reshape(batch_x * self.max_q_per_link, -1).unsqueeze(0).expand(batch_q,
                                                                                                               -1, -1)
            q_batch_temp = q_batch[:, :i + 1].unsqueeze(1).expand(-1, batch_x * self.max_q_per_link, -1)
            d_norm = torch.norm((q_batch_temp - q_lib_temp), dim=-1).reshape(batch_q, batch_x, self.max_q_per_link)

            d_norm_min, d_norm_min_idx = d_norm.min(dim=-1)
            grad = torch.autograd.grad(d_norm_min.reshape(-1), q_batch_temp, torch.ones_like(d_norm_min.reshape(-1)),
                                       retain_graph=True)[0]
            grad_min_q = grad.reshape(batch_q, batch_x, self.max_q_per_link, -1).gather(2, d_norm_min_idx.unsqueeze(
                -1).unsqueeze(-1).expand(-1, -1, -1, i + 1))[:, :, 0, :]
            grad_tensor[:, :, :i + 1, i] = grad_min_q.transpose(0, 1)
            d_tensor[:, :, i] = d_norm_min.transpose(0, 1)

        d, d_min_idx = d_tensor.min(dim=-1)
        grad_final = grad_tensor.gather(3, d_min_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 7, 7))[:, :, :, 0]
        return d, grad_final

    def sample_q(self, batch_q=None):
        if batch_q is None:
            batch_q = self.batch_q
        q_sampled = self.panda.theta_min + torch.rand(batch_q, 7).to(self.device) * (
                self.panda.theta_max - self.panda.theta_min)
        q_sampled.requires_grad = True
        return q_sampled

    def projection(self, q, d, grad):
        q_new = q - grad * d.unsqueeze(-1)
        return q_new

    def check_data(self):
        # x_batch:(batch_x,3)
        # q_batch:(batch_q,7)
        # d:(batch_x,batch_q)
        # grad:(batch_x,batch_q,7)
        x_batch, q_batch, d, grad = self.select_data()
        q_proj = self.projection(q_batch, d, grad)

        # visualize
        import trimesh
        pose = torch.eye(4).unsqueeze(0).to(self.device).float()
        for q0, q1 in zip(q_batch, q_proj[1]):
            scene = trimesh.Scene()
            scene.add_geometry(trimesh.PointCloud(x_batch.data.cpu().numpy(), colors=[255, 0, 0]))
            robot_mesh0 = self.panda.get_forward_robot_mesh(pose, q0.unsqueeze(0))[0]
            robot_mesh0 = np.sum(robot_mesh0)
            robot_mesh0.visual.face_colors = [0, 255, 0, 100]
            scene.add_geometry(robot_mesh0)
            robot_mesh1 = self.panda.get_forward_robot_mesh(pose, q1.unsqueeze(0))[0]
            robot_mesh1 = np.sum(robot_mesh1)
            robot_mesh1.visual.face_colors = [0, 0, 255, 100]
            scene.add_geometry(robot_mesh1)
            scene.show()

    def load_np_csdf_data(self):
        # resize the data with grids
        x_new = torch.cat([torch.from_numpy(self.raw_data[k]['x']).unsqueeze(0) for k in self.raw_data.keys()],
                          dim=0).to(self.device)
        q_new = torch.ones(8000, self.max_q_per_link, 7, 7).to(self.device) * torch.inf
        q_new[self.data['k']] = self.data['q']
        np_csdf_data = {
            'x': x_new,
            'q': q_new,
        }
        return np_csdf_data

    def np_csdf(self, x, q, visualize=False):
        # Non parametric CS-DF
        # x : (Nx,3)
        # q : (Nq,7)

        assert (x[:, 0] < 0.5).all() and (x[:, 0] >= -0.5).all()
        assert (x[:, 1] < 0.5).all() and (x[:, 1] >= -0.5).all()
        assert (x[:, 2] < 1.0).all() and (x[:, 2] >= 0.0).all()

        q.requires_grad = True

        # write a function to find the index of voxel grid.
        # workspace: [[-0.5,0.5],[-0.5,0.5],[0.,1.0]], data resolution: 1/(20-1)
        x_norm = x - torch.tensor([[-0.5, -0.5, 0]]).to(self.device)  # normalize x to [0,1)

        x_idx = (x_norm * 19).long()
        # Define the eight possible offsets for the neighbors
        offsets = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1, ], [0, 1, 1], [1, 1, 1]]).to(self.device)

        # Compute the eight neighbors by adding the offsets to the voxel indices
        neighbor_idx = (x_idx.unsqueeze(1) + offsets.unsqueeze(0))
        x_grid = neighbor_idx / 19.0 + torch.tensor([[-0.5, -0.5, 0]]).to(self.device)
        x_dist = torch.norm(x_grid - x.unsqueeze(1), dim=-1).unsqueeze(-1).expand(-1, -1, len(q))

        data_key = neighbor_idx[:, :, 0] * 20 * 20 + neighbor_idx[:, :, 1] * 20 + neighbor_idx[:, :, 2]
        data_key_unique, data_key_unique_idx = torch.unique(data_key, return_inverse=True)
        data_key_unique_idx = data_key_unique_idx.reshape(-1)
        q_lib_unique = self.np_csdf_data['q'][data_key_unique]
        d, g = self.decode_distance(q, q_lib_unique)

        d_expand = d.unsqueeze(0).expand(len(data_key_unique_idx), -1, -1)
        g_expand = g.unsqueeze(0).expand(len(data_key_unique_idx), -1, -1, -1)
        d = torch.gather(d_expand, 1, data_key_unique_idx.unsqueeze(-1).unsqueeze(-1).expand(d_expand.shape))[:, 0, :]
        g = torch.gather(g_expand, 1,
                         data_key_unique_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(g_expand.shape))[:, 0, :,
            :]
        d, g = d.reshape(len(x), 8, len(q)), g.reshape(len(x), 8, len(q), 7)
        d, min_idx_d = torch.min(d, dim=1)
        # weights = torch.where(d.isinf(),0,x_dist)
        # weights = torch.nn.functional.normalize(weights,dim=1,p=1)
        # d = torch.where(d.isinf(),100,d)
        # d = torch.sum(d*weights,dim=1)
        # g = torch.sum(g*weights.unsqueeze(-1),dim=1)
        # g = torch.nn.functional.normalize(g,dim=-1,p=2)
        g = g.gather(1, min_idx_d.unsqueeze(1).unsqueeze(-1).expand(-1, 8, -1, 7))[:, 0, :, :]

        if visualize:
            q_proj = self.projection(q, d, g)
            print(q.shape, q_proj.shape)
            import trimesh
            pose = torch.eye(4).unsqueeze(0).to(self.device).float()
            for q0, q1 in zip(q, q_proj[0]):
                print(q0, q1)
                scene = trimesh.Scene()
                scene.add_geometry(trimesh.PointCloud(x.data.cpu().numpy(), colors=[255, 0, 0]))
                scene.add_geometry(trimesh.PointCloud(x_grid.reshape(-1, 3).data.cpu().numpy(), colors=[0, 255, 0]))
                robot_mesh0 = self.panda.get_forward_robot_mesh(pose, q0.unsqueeze(0))[0]
                robot_mesh0 = np.sum(robot_mesh0)
                robot_mesh0.visual.face_colors = [0, 255, 0, 100]
                scene.add_geometry(robot_mesh0)
                robot_mesh1 = self.panda.get_forward_robot_mesh(pose, q1.unsqueeze(0))[0]
                robot_mesh1 = np.sum(robot_mesh1)
                robot_mesh1.visual.face_colors = [0, 0, 255, 100]
                scene.add_geometry(robot_mesh1)
                scene.show()

        return d, g

    def train_nn(self, epoches=500):

        # model
        # input: [x,q] (B,3+7)

        model = MLPRegression(input_dims=10, output_dims=1, mlp_layers=[1024, 512, 256, 128, 128], skips=[],
                              act_fn=torch.nn.ReLU, nerf=True)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5000,
                                                               threshold=0.01, threshold_mode='rel',
                                                               cooldown=0, min_lr=0, eps=1e-04, verbose=True)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        COSLOSS = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        model_dict = {}
        for iter in range(epoches):
            model.train()
            with torch.cuda.amp.autocast():
                x_batch, q_batch, d, gt_grad = self.select_data()

                x_inputs = x_batch.unsqueeze(1).expand(-1, self.batch_q, -1).reshape(-1, 3)
                q_inputs = q_batch.unsqueeze(0).expand(self.batch_x, -1, -1).reshape(-1, 7)

                inputs = torch.cat([x_inputs, q_inputs], dim=-1)
                outputs = d.reshape(-1, 1)
                gt_grad = gt_grad.reshape(-1, 7)

                weights = torch.ones_like(outputs).to(device)
                # weights = (1/outputs).clamp(0,1)

                d_pred = model.forward(inputs)
                d_grad_pred = \
                    torch.autograd.grad(d_pred, q_inputs, torch.ones_like(d_pred), retain_graph=True,
                                        create_graph=True)[0]

                # Compute the Eikonal loss
                eikonal_loss = torch.abs(d_grad_pred.norm(2, dim=-1) - 1).mean()

                # Compute the tension loss
                dd_grad_pred = torch.autograd.grad(d_grad_pred, q_inputs, torch.ones_like(d_pred), retain_graph=True,
                                                   create_graph=True)[0]

                # gradient loss
                gradient_loss = (1 - COSLOSS(d_grad_pred, gt_grad)).mean()
                # tension loss
                tension_loss = dd_grad_pred.square().sum(dim=-1).mean()
                # Compute the MSE loss
                d_loss = ((d_pred - outputs) ** 2 * weights).mean()

                # Combine the two losses with appropriate weights
                w0 = 5.0
                w1 = 0.01
                w2 = 0.01
                w3 = 0.1
                loss = w0 * d_loss + w1 * eikonal_loss + w2 * tension_loss + w3 * gradient_loss

                # # Print the losses for monitoring

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step(loss)
                if iter % 10 == 0:
                    print(
                        f"Epoch:{iter}\tMSE Loss: {d_loss.item():.3f}\tEikonal Loss: {eikonal_loss.item():.3f}\tTension Loss: {tension_loss.item():.3f}\tGradient Loss: {gradient_loss.item():.3f}\tTotal loss:{loss.item():.3f}\tTime: {time.strftime('%H:%M:%S', time.gmtime())}")
                    model_dict[iter] = model.state_dict()
                    torch.save(model_dict, os.path.join(CUR_PATH, 'model_dict.pt'))
        return model

    def inference(self, x, q, model):
        model.eval()
        x, q = x.to(self.device), q.to(self.device)
        # q.requires_grad = True
        x_cat = x.unsqueeze(1).expand(-1, len(q), -1).reshape(-1, 3)
        q_cat = q.unsqueeze(0).expand(len(x), -1, -1).reshape(-1, 7)
        inputs = torch.cat([x_cat, q_cat], dim=-1)
        sdf_pred = model.forward(inputs)
        return sdf_pred

    def inference_d_wrt_q(self, x, q, model, return_grad=True):
        sdf_pred = self.inference(x, q, model)
        d = sdf_pred.reshape(len(x), len(q)).min(dim=0)[0]
        if return_grad:
            grad = torch.autograd.grad(d, q, torch.ones_like(d), retain_graph=True, create_graph=True)[0]
            # dgrad = torch.autograd.grad(grad,q,torch.ones_like(grad),retain_graph=True,create_graph=True)[0]
            return d, grad
        else:
            return d

    def inference_sdf(self, p, q):
        pose = torch.eye(4).unsqueeze(0).expand(len(q), -1, -1).to(self.device).float()
        sdf, _ = self.bp_sdf.get_whole_body_sdf_batch(p, pose, q, self.bp_sdf_model, use_derivative=False)
        d = sdf.min()
        grad = torch.autograd.grad(d, q, torch.ones_like(d), retain_graph=True, create_graph=True)[0]
        return d, grad

    def inference_njsdf(self, x, q):

        q = q.unsqueeze(1).expand(len(q), len(x), 7)
        x = x.unsqueeze(0).expand(len(q), len(x), 3)
        x_cat = torch.cat([q, x], dim=-1).float().reshape(-1, 10)
        dist = self.njsdf_model.forward(x_cat) / 100.
        d = (torch.min(dist[:, :8], dim=-1)[0])
        grad = torch.autograd.grad(d, q, torch.ones_like(d), retain_graph=True, create_graph=True)[0]
        min_d, idx = torch.min(d, dim=0)
        grad = grad.squeeze(0)[idx]
        return min_d, grad.unsqueeze(0)

    def check_nn(self, model):
        # x,q,d_gt,grad_gt = self.select_data()
        x = torch.rand(100, 3).to(device) - torch.tensor([[0.5, 0.5, 0]]).to(device)
        q = self.sample_q()
        q_proj = q
        for i in range(1):
            t0 = time.time()
            d, grad = self.inference_d_wrt_q(x, q_proj, model)
            q_proj = self.projection(q_proj, d, grad)
            print(f'iter {i} finished, time cost: {time.time() - t0}')

            # q_proj = self.projection(q,d_gt.squeeze(0),grad_gt.squeeze(0))
        # visualize
        import trimesh
        pose = torch.eye(4).unsqueeze(0).to(self.device).float()
        for q0, q1, d, g in zip(q, q_proj, d, grad):
            print(d, g)
            scene = trimesh.Scene()
            scene.add_geometry(trimesh.PointCloud(x.data.cpu().numpy(), colors=[255, 0, 0]))
            robot_mesh0 = self.panda.get_forward_robot_mesh(pose, q0.unsqueeze(0))[0]
            robot_mesh0 = np.sum(robot_mesh0)
            robot_mesh0.visual.face_colors = [0, 255, 0, 100]
            scene.add_geometry(robot_mesh0)
            robot_mesh1 = self.panda.get_forward_robot_mesh(pose, q1.unsqueeze(0))[0]
            robot_mesh1 = np.sum(robot_mesh1)
            robot_mesh1.visual.face_colors = [0, 0, 255, 100]
            scene.add_geometry(robot_mesh1)
            scene.show()

    def eval_nn(self, model):
        eval_time = False
        eval_acc = True
        if eval_time:
            x = torch.rand(100, 3).to(device) - torch.tensor([[0.5, 0.5, 0]]).to(device)
            q = self.sample_q(batch_q=100)
            time_cost_list = []
            for i in range(100):
                t0 = time.time()
                d = self.inference_d_wrt_q(x, q, model, return_grad=False)
                t1 = time.time()
                grad = torch.autograd.grad(d, q, torch.ones_like(d), retain_graph=True, create_graph=True)[0]
                q_proj = self.projection(q, d, grad)
                t2 = time.time()
                if i > 0:
                    time_cost_list.append([t1 - t0, t2 - t1])
            mean_time_cost = np.mean(time_cost_list, axis=0)
            print(f'inference time cost:{mean_time_cost[0]}\t projection time cost: {mean_time_cost[1]}')

        if eval_acc:
            # bp_sdf model
            bp_sdf = bf_sdf.BPSDF(8, -1.0, 1.0, self.panda, device)
            bp_sdf_model = torch.load(os.path.join(CUR_PATH, '../RDF/models/BP_8.pt'))

            res = []
            for i in range(1000):
                x = torch.rand(1, 3).to(device) - torch.tensor([[0.5, 0.5, 0]]).to(device)
                q = self.sample_q(batch_q=1000)
                for _ in range(1):
                    d, grad = self.inference_d_wrt_q(x, q, model, return_grad=True)
                    q = self.projection(q, d, grad)
                q, grad = q.detach(), grad.detach()  # release memory
                pose = torch.eye(4).unsqueeze(0).expand(len(q), -1, -1).to(self.device).float()
                sdf, _ = bp_sdf.get_whole_body_sdf_batch(x, pose, q, bp_sdf_model, use_derivative=False)

                error = sdf.reshape(-1).abs()
                MAE = error.mean()
                RMSE = torch.sqrt(torch.mean(error ** 2))
                SR = (error < 0.03).sum().item() / len(error)
                res.append([MAE.item(), RMSE.item(), SR])
                print(f'iter {i} finished, MAE:{MAE}\tRMSE:{RMSE}\tSR:{SR}')
            res = np.array(res)
            print(f'MAE:{res[:, 0].mean()}\tRMSE:{res[:, 1].mean()}\tSR:{res[:, 2].mean()}')


def wall(size, center, rot):
    # center: (3,)
    # size: (3,)
    # return: (N,3)
    # the gap of the linspace is 0.1
    x = torch.arange(-size[0] / 2, size[0] / 2, 0.1).to(device)
    y = torch.arange(-size[1] / 2, size[1] / 2, 0.1).to(device)
    x, y = torch.meshgrid(x, y)
    x, y = x.reshape(-1), y.reshape(-1)
    z = torch.zeros_like(x).to(device)
    points = torch.stack([x, y, z], dim=-1)
    points = torch.matmul(points, rot.transpose(0, 1)) + center
    return points


def ring(radius, center, rot):
    # center: (3,)
    # size: (3,)
    # return: (N,3)
    # the gap of the linspace is 0.1
    theta = torch.arange(0, 2 * PI, 0.4).to(device)
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    z = torch.zeros_like(x).to(device)
    points = torch.stack([x, y, z], dim=-1)
    points = torch.matmul(points, rot.transpose(0, 1)) + center
    return points


def mannully_choose_q():
    q = cdf.panda.theta_min + torch.rand(100, 7).to(cdf.device) * (cdf.panda.theta_max - cdf.panda.theta_min)
    pose = torch.eye(4).unsqueeze(0).to(device).float()
    robot_mesh = []
    for _q in q:
        rm = cdf.panda.get_forward_robot_mesh(pose, _q.unsqueeze(0))[0]
        mesh = np.sum(rm)
        mesh.visual.face_colors = [0, 255, 0, 100]
        robot_mesh.append(mesh)
    return q, robot_mesh


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    cdf = CDF(device)

    # cdf.np_csdf(torch.rand(100,3).to(device) - torch.tensor([[0.5,0.5,0.0]]).to(device),torch.rand(10,7).to(device),visualize=False)

    # trainer.train_nn(epoches=20000)
    model = MLPRegression(input_dims=10, output_dims=1, mlp_layers=[1024, 512, 256, 128, 128], skips=[],
                          act_fn=torch.nn.ReLU, nerf=True)
    # model.load_state_dict(torch.load(os.path.join(CUR_PATH,'model_dict.pt'))[19900])
    model.load_state_dict(torch.load(os.path.join(CUR_PATH, 'model_dict_tension_2.pt'))[49900])
    model.to(device)

    wall_size = torch.tensor([0.5, 0.5]).to(device)
    wall_center = torch.tensor([0.5, 0.0, 0.2]).to(device)
    wall_rot = torch.tensor([[1.0, 0.0, 0.0],
                             [0.0, 0.0, -1.0],
                             [0.0, 1.0, 0.0], ]).to(device)

    # p = wall(wall_size,wall_center,wall_rot)

    ring_radius = 0.2
    ring_center = torch.tensor([0.4, 0.0, 0.45]).to(device)
    ring_rot = torch.tensor([[0.0, 0.0, -1.0],
                             [0.0, 1.0, 0.0],
                             [1.0, 0.0, 0.0], ]).to(device)
    p = ring(ring_radius, ring_center, ring_rot)

    # # define a sphere
    # sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
    # sphere.visual.face_colors = [255,0,0,100]
    # sphere_center = [0.3, 0.3, 0.4]
    # sphere.apply_translation(sphere_center)

    # points = sphere.vertices
    # print(f'sampled points on sphere: {points.shape}')
    # q = torch.tensor([[ 0.0291, -1.4924,  1.3225, -2.7588, -2.5121,  3.5030,  1.5731]]).to(device).float()
    # q = torch.tensor([[-1.8044, -1.3162, -1.5007, -1.0240, -1.7643,  0.8293, -1.1870]]).to(device).float()

    # wall
    # q = torch.tensor([[-0.03610672,  0.14759123,  0.60442339, -2.45172895, -0.06231244,
    #     2.53993935,  1.10256184]]).to(device).float()
    # q0 = torch.tensor([[-0.25802498, -0.01593395, -0.35283275, -2.24489454, -0.06160258,
    #     2.35934126,  0.34169443]]).to(device).float()

    # ringl
    q = torch.tensor([[-0.41302193, -0.94202107, -0.32044236, -2.47843634, -0.16166646,
                       1.74380392, 0.75803596]]).to(device).float()
    q0 = torch.tensor([[-0.23643478, -0.06273081, 0.16822745, -2.54468149, -0.32664142,
                        3.57070459, 1.14682636]]).to(device).float()
    q = q0

    q.requires_grad = True
    d, grad = cdf.inference_d_wrt_q(p, q, model, return_grad=True)
    q_proj = cdf.projection(q, d, grad)
    print(d, grad)

    scene = trimesh.Scene()
    for p0 in p.data.cpu().numpy():
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
        sphere.visual.face_colors = [255, 0, 0, 100]
        sphere.apply_translation(p0)
        scene.add_geometry(sphere)

    pose = torch.eye(4).unsqueeze(0).to(device).float()
    rm = cdf.panda.get_forward_robot_mesh(pose, q)[0]
    mesh = np.sum(rm)
    mesh.visual.face_colors = [0, 255, 0, 100]
    scene.add_geometry(mesh)
    scene.show()

    # q,robot_mesh = mannully_choose_q()
    # for _q,m in zip(q,robot_mesh):
    #     print(_q)
    #     scene.add_geometry(m,'mesh')
    #     scene.show()
    #     scene.delete_geometry('mesh')
