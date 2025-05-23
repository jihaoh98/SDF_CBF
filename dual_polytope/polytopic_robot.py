import numpy as np
from math import cos, sin
import cvxpy as cp
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


class Polytopic_Robot:
    def __init__(self, indx, init_state, **kwargs) -> None:
        """ Init the polytopic robot """
        self.id = indx
        self.init_state = init_state
        self.cur_state = None
        self.A0 = None
        self.b0 = None
        self.G0 = None
        self.g0 = None
        self.vertices = None
        self.obs_vertices = None


if __name__ == "__main__":
    pass