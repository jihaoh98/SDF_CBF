import copy
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch


class Circle:
    def __init__(self, center, radius, device='cpu', label=None):
        self.center = center.to(device)
        self.radius = radius
        self.device = device
        self.label = label

    def signed_distance(self, p):
        # p: N x 2, 2D points input
        # return: N x 1, signed distance output

        N = p.size(0)  # the number of points
        return (torch.norm(p - self.center.unsqueeze(0).expand(N, -1), dim=1) - self.radius).unsqueeze(-1)

    def signed_distance2(self, p):
        # p: N x 2, 2D points input
        # return: N x 1, signed distance output

        N = p.size(0)  # the number of points
        dist = (torch.norm(p - self.center.unsqueeze(0).expand(N, -1), dim=1) - self.radius).unsqueeze(-1)
        center = self.center.unsqueeze(0).expand(N, -1)
        return dist, center

    def normal(self, p):
        # p: N x 2, 2D points input
        # return: N x 2, normal vector output

        d = self.signed_distance(p)
        grad = torch.autograd.grad(d.sum(), p, create_graph=True, retain_graph=True)[0]
        n = torch.nn.functional.normalize(grad, dim=-1)  # normalize the gradient to get the normal vector
        return n

    def sample_surface(self, N):
        theta = torch.rand(N, 1).to(self.device) * 2 * math.pi
        x = torch.cat([torch.cos(theta).to(self.device), torch.sin(theta).to(self.device)], dim=-1)
        return x * self.radius + self.center.unsqueeze(0).expand(N, -1)

    def create_patch(self, color='k'):
        center = self.center.detach().cpu().numpy()
        radius = self.radius
        circle = patches.Circle(center, radius, linewidth=1, edgecolor=color, facecolor='none', hatch='///',
                                label=self.label)
        return circle


class Box:
    def __init__(self, center, w, h, device='cpu', label=None):
        self.center = center
        self.w = w
        self.h = h
        self.device = device
        self.label = label

    def signed_distance(self, p):
        N = p.size(0)
        p = p - self.center.unsqueeze(0).expand(N, -1)
        d = torch.abs(p) - torch.tensor([self.w / 2., self.h / 2.]).to(self.device).unsqueeze(0).expand(N, -1)
        dist = torch.norm(torch.nn.functional.relu(d), dim=1) + \
               torch.min(torch.cat([torch.max(d, dim=1)[0].unsqueeze(1), torch.zeros(N, 1).to(self.device)], dim=1),
                         dim=1)[0]

        return dist.unsqueeze(-1)

    def sample_surface(self, N):
        N_per_edge = N // 4
        remainder = N % 4  # Extra samples for the adjustment

        # Generate points for each edge
        x = torch.linspace(-self.w / 2, self.w / 2, steps=N_per_edge + 1, device=self.device)[
            :-1]  # Avoid duplicate at corner
        y = torch.linspace(-self.h / 2, self.h / 2, steps=N_per_edge + 1, device=self.device)[:-1]  # Same here

        # Concatenating points from all edges
        top_edge = torch.stack([x, torch.full_like(x, self.h / 2)], dim=1)
        bottom_edge = torch.stack([x, torch.full_like(x, -self.h / 2)], dim=1)
        left_edge = torch.stack([torch.full_like(y, -self.w / 2), y], dim=1)
        right_edge = torch.stack([torch.full_like(y, self.w / 2), y], dim=1)

        # Combine all edges
        all_points = torch.cat([top_edge, bottom_edge, left_edge, right_edge], dim=0)

        if remainder > 0:
            # Add extra points to one of the sides, for example, to the top
            extra_x = torch.linspace(-self.w / 2, self.w / 2, steps=remainder, device=self.device)
            extra_top_edge = torch.stack([extra_x, torch.full_like(extra_x, self.h / 2)], dim=1)
            all_points = torch.cat([all_points, extra_top_edge], dim=0)

        # Adjust points relative to the center
        all_points += self.center.unsqueeze(0)

        return all_points

    def normal(self, p):
        d = self.signed_distance(p)
        grad = torch.autograd.grad(d.sum(), p, create_graph=True, retain_graph=True)[0]
        n = torch.nn.functional.normalize(grad, dim=-1)
        return n

    def create_patch(self, color='r'):
        center = self.center.cpu().numpy()
        w = self.w
        h = self.h
        if self.label is None:
            rect = patches.Rectangle(center - [w / 2.0, h / 2.0], w, h, linewidth=1, edgecolor=color, facecolor='none',
                                     hatch='///')
        else:
            rect = patches.Rectangle(center - [w / 2.0, h / 2.0], w, h, linewidth=1, edgecolor=color, facecolor='none',
                                     hatch='///', label=self.label)
        return rect


class Triangle:
    def __init__(self, p0, p1, p2):
        self.p0 = torch.tensor(p0, dtype=torch.float)
        self.p1 = torch.tensor(p1, dtype=torch.float)
        self.p2 = torch.tensor(p2, dtype=torch.float)

    def signed_distance(self, p):
        e0 = self.p1 - self.p0
        e1 = self.p2 - self.p1
        e2 = self.p0 - self.p2
        v0 = p - self.p0
        v1 = p - self.p1
        v2 = p - self.p2
        pq0 = v0 - e0 * torch.clamp(torch.dot(v0, e0) / torch.dot(e0, e0), 0, 1)
        pq1 = v1 - e1 * torch.clamp(torch.dot(v1, e1) / torch.dot(e1, e1), 0, 1)
        pq2 = v2 - e2 * torch.clamp(torch.dot(v2, e2) / torch.dot(e2, e2), 0, 1)
        s = torch.sign((e0[0] * e2[1] - e0[1] * e2[0]))

        d = torch.min(torch.min(
            torch.cat([torch.dot(pq0, pq0).unsqueeze(0), s * (v0[0] * e0[1] - v0[1] * e0[0]).unsqueeze(0)], dim=0),
            torch.cat([torch.dot(pq1, pq1).unsqueeze(0), s * (v1[0] * e1[1] - v1[1] * e1[0]).unsqueeze(0)], dim=0)),
            torch.cat([torch.dot(pq2, pq2).unsqueeze(0), s * (v2[0] * e2[1] - v2[1] * e2[0]).unsqueeze(0)],
                      dim=0))
        return -torch.sqrt(d[0]) * torch.sign(d[1])

    def normal(self, p):
        d = self.signed_distance(p)
        grad = torch.autograd.grad(d, p, create_graph=True, retain_graph=True)[0]
        n = torch.nn.functional.normalize(grad, dim=-1)
        return n

    def create_patch(self, color='g'):
        p0 = self.p0.numpy()
        p1 = self.p1.numpy()
        p2 = self.p2.numpy()
        vertices = [p0, p1, p2]
        triangle = patches.Polygon(vertices, linewidth=1, edgecolor=color, facecolor=color)
        return triangle


class Ellipse:
    def __init__(self, center, ab):
        self.center = torch.tensor(center)
        self.ab = torch.tensor(ab)

    def signed_distance(self, p):
        e = 1e-8
        p = p - self.center
        p = torch.abs(p)
        if p[0] > p[1]:
            n_p = torch.flip(p, [0])
            n_ab = torch.flip(self.ab, [0])
        else:
            n_p = p
            n_ab = self.ab
        l = n_ab[1] * n_ab[1] - n_ab[0] * n_ab[0]
        m = n_ab[0] * n_p[0] / (l + e)
        n = n_ab[1] * n_p[1] / (l + e)
        m2 = m * m
        n2 = n * n
        c = (m2 + n2 - 1.0) / 3.0
        c3 = c * c * c
        q = c3 + m2 * n2 * 2.0
        d = c3 + m2 * n2
        g = m + m * n2
        co = None
        if d < 0.0:
            h = torch.acos(q / (c3 - e)) / 3.0
            s = torch.cos(h)
            t = torch.sin(h) * (3.0 ** 0.5)
            rx = torch.sqrt(-c * (s + t + 2.0) + m2)
            ry = torch.sqrt(-c * (s - t + 2.0) + m2)
            co = (ry + torch.sign(l) * rx + torch.abs(g) / (rx * ry + e) - m) / 2.0
        else:
            h = 2.0 * m * n * math.sqrt(d)
            s = torch.sign(q + h) * torch.pow(torch.abs(q + h), 1.0 / 3.0)
            u = torch.sign(q - h) * torch.pow(torch.abs(q - h), 1.0 / 3.0)
            rx = -s - u - c * 4.0 + 2.0 * m2
            ry = (s - u) * math.sqrt(3.0)
            rm = torch.sqrt(rx * rx + ry * ry)
            co = (ry / (torch.sqrt(rm - rx) + e) + 2.0 * g / (rm + e) - m) / 2.0
        r = n_ab * torch.tensor([co, torch.sqrt(1.0 - co * co)])
        return torch.sqrt(torch.sum((r - n_p) ** 2)) * torch.sign(n_p[1] - r[1])

    def normal(self, p):
        d = self.signed_distance(p)
        grad = torch.autograd.grad(d.sum(), p, create_graph=True, retain_graph=True)[0]
        n = torch.nn.functional.normalize(grad, dim=-1)
        return n

    def create_patch(self, color='g'):
        center = self.center.numpy()
        ab = self.ab.numpy()
        ellipse = patches.Ellipse(center, ab[0] * 2, ab[1] * 2, angle=0, linewidth=1, edgecolor=color, facecolor=color)
        return ellipse


# Operators
class Union:
    def __init__(self, a, *bs, k=None):
        self.a = a
        self.bs = bs
        self.k = k

    def signed_distance(self, p):
        def f(p):
            d1 = self.a.signed_distance(p)
            for i, b in enumerate(self.bs):
                d2 = b.signed_distance(p)
                K = self.k[i]
                if K is None:
                    d1 = torch.min(d1, d2)
                else:
                    h = torch.clamp(0.5 + 0.5 * (d2 - d1) / K, 0, 1)
                    m = d2 + (d1 - d2) * h
                    d1 = m - K * h * (1 - h)
            return d1

        return f(p)

    def normal(self, p):
        d = self.signed_distance(p)
        grad = torch.autograd.grad(d.sum(), p, create_graph=True, retain_graph=True)[0]
        n = torch.nn.functional.normalize(grad, dim=-1)
        return n

    def create_patch(self):
        pass


class Difference:
    def __init__(self, a, *bs, k=None):
        self.a = a
        self.bs = bs
        self.k = k

    def signed_distance(self, p):
        def f(p):
            d1 = self.a.signed_distance(p)
            for i, b in enumerate(self.bs):
                d2 = b.signed_distance(p)
                K = self.k[i]
                if K is None:
                    d1 = torch.max(d1, -d2)
                else:
                    h = torch.clamp(0.5 - 0.5 * (d2 + d1) / K, 0, 1)
                    m = d1 + (-d2 - d1) * h
                    d1 = m + K * h * (1 - h)
            return d1

        return f(p)

    def create_patch(self):
        pass


class Intersection:
    def __init__(self, a, *bs, k=None):
        self.a = a
        self.bs = bs
        self.k = k

    def signed_distance(self, p):
        def f(p):
            d1 = self.a.signed_distance(p)
            for i, b in enumerate(self.bs):
                d2 = b.signed_distance(p)
                K = self.k[i]
                if K is None:
                    d1 = torch.max(d1, d2)
                else:
                    h = torch.clamp(0.5 - 0.5 * (d2 - d1) / K, 0, 1)
                    m = d2 + (d1 - d2) * h
                    d1 = m + K * h * (1 - h)
            return d1

        return f(p)

    def create_patch(self):
        pass


class Blend:
    def __init__(self, a, *bs, k=[0.5]):
        self.a = a
        self.bs = bs
        self.k = k

    def signed_distance(self, p):
        def f(p):
            d1 = self.a.signed_distance(p)
            for i, b in enumerate(self.bs):
                d2 = b.signed_distance(p)
                K = self.k[i]
                d1 = K * d2 + (1 - K) * d1
            return d1

        return f(p)

    def create_patch(self):
        pass


class Negate:
    def __init__(self, shape):
        self.shape = shape

    def signed_distance(self, p):
        def f(p):
            return -self.shape.signed_distance(p)

        return f(p)


class Dilate:
    def __init__(self, shape, r):
        self.shape = shape
        self.r = r

    def signed_distance(self, p):
        def f(p):
            return self.shape.signed_distance(p) - self.r

        return f(p)


class Erode:
    def __init__(self, shape, r):
        self.shape = shape
        self.r = r

    def signed_distance(self, p):
        def f(p):
            return self.shape.signed_distance(p) + self.r

        return f(p)


class Shell:
    def __init__(self, shape, thickness):
        self.shape = shape
        self.thickness = thickness

    def signed_distance(self, p):
        def f(p):
            return torch.abs(self.shape.signed_distance(p)) - self.thickness / 2

        return f(p)


def test_circle():
    center = torch.tensor([0, 0])
    radius = 1
    circle = Circle(center=center, radius=radius)
    p = torch.tensor([[0, 0], [1, 0], [0, 1], [0.5, 0.5]], requires_grad=True)
    d = circle.signed_distance(p)
    print(d)

    # fig, ax = plt.subplots()
    # ax.set_xlim(-2, 2)
    # ax.set_ylim(-2, 2)
    # ax.add_patch(circle.create_patch())
    # ax.scatter(p[:,0].detach().numpy(),p[:,1].detach().numpy())
    # ax.axis('equal')
    # plt.show()

    # test normal
    # n = circle.normal(p)
    # print(n)

    # fig, ax = plt.subplots()
    # ax.set_xlim(-2, 2)
    # ax.set_ylim(-2, 2)
    # ax.add_patch(circle.create_patch())
    # ax.scatter(p[:,0].detach().numpy(),p[:,1].detach().numpy())
    # ax.quiver(p[:,0].detach().numpy(),p[:,1].detach().numpy(),n[:,0].detach().numpy(),n[:,1].detach().numpy())
    # ax.axis('equal')
    # plt.show()

    # test sample_surface
    N = 100
    p = circle.sample_surface(N)

    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.add_patch(circle.create_patch())
    ax.scatter(p[:, 0].detach().numpy(), p[:, 1].detach().detach().numpy())
    ax.axis('equal')
    plt.show()


def test_box():
    center = torch.tensor([0, 0])
    w = 1
    h = 1
    box = Box(center=center, w=w, h=h)
    p = torch.tensor([[0, 0], [1, 0], [0, 1], [0.5, 0.5]], requires_grad=True)
    d = box.signed_distance(p)
    print(d)

    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.add_patch(box.create_patch())
    ax.scatter(p[:, 0].detach().numpy(), p[:, 1].detach().numpy())
    ax.axis('equal')
    plt.show()

    # # test normal
    # n = box.normal(p)
    # print(n)

    # fig, ax = plt.subplots()
    # ax.set_xlim(-2, 2)
    # ax.set_ylim(-2, 2)
    # ax.add_patch(box.create_patch())
    # ax.scatter(p[:,0].detach().numpy(),p[:,1].detach().numpy())
    # ax.quiver(p[:,0].detach().numpy(),p[:,1].detach().numpy(),n[:,0].detach().numpy(),n[:,1].detach().numpy())
    # ax.axis('equal')
    # plt.show()

    # test sample_surface
    N = 50
    p = box.sample_surface(N)

    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.add_patch(box.create_patch())
    ax.scatter(p[:, 0].detach().numpy(), p[:, 1].detach().detach().numpy())
    ax.axis('equal')
    plt.show()


if __name__ == "__main__":
    test_circle()
    # test_box()
