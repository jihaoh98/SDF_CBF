import numpy as np
import yaml
from math import sqrt


class Circle_Obs:
    def __init__(self, index, radius, center, vel=np.zeros((2, )), goal=np.zeros((2, 1)), mode='static', **kwargs) -> None:
        """ init the circular-shaped obstacle, index to distinguish different obstacles """
        self.id = index
        self.radius = radius
        self.init_state = np.array(center)
        self.state = np.copy(self.init_state)

        self.vel = np.array(vel)
        self.goal = goal
        self.arrive_flag = False
        self.mode = mode

    def is_collision(self, point):
        """ determine if the point has collision with the circle obstacle """
        distance = (self.state[0] - point[0]) ** 2 + (self.state[1] - point[1]) ** 2
        distance = sqrt(distance)

        if distance >= self.radius:
            return False
        else:
            return True

    def arrive_destination(self):
        """ determine if the robot arrives its goal position """
        dist = np.linalg.norm(self.state.reshape(2, 1) - self.goal[0:2])

        if dist < 0.1:
            self.arrive_flag = True
            self.vel = np.zeros((2, ))
            return True
        else:
            self.arrive_flag = False
            return False

    def move_forward(self, step_time):
        """ move this obstacle if its model is dynamic"""
        if self.mode != 'static':
            if self.arrive_flag:
                return
            self.state = self.state + self.vel * step_time
            if not self.arrive_flag:
                self.arrive_destination()

    def get_current_state(self):
        """ return the obstacle's position and velocity, as well as radius """
        current_state = np.array([self.state[0], self.state[1], self.vel[0], self.vel[1], self.radius])
        return current_state
    

class Polytopic_Obs:
    def __init__(self, index, vertex=None, vel=np.zeros((2, )), goal=np.zeros((2, 1)), mode='static', **kwargs) -> None:
        """ init the polytopic-shaped obstacle """
        self.vertexes = None
        self.init_vertexes = None
        self.ver_num = None

        if vertex is not None:
            # in shape (n, 2)
            self.init_vertexes = np.array(vertex, ndmin=2)
            self.vertexes = np.copy(self.init_vertexes)
            self.ver_num = self.vertexes.shape[0]

        self.id = index
        self.init_position = None
        self.position = None
        self.vertex_vertor = None

        self.vel = np.array(vel)
        self.goal = goal
        self.arrive_flag = False

        self.mode = mode
        self.edge_list = None

        # collision check
        self.A = None
        self.b = None

        # init
        self.get_center_position()
        self.get_vertex_vector()
        self.get_edges()
        self.get_matrix()

    def get_center_position(self):
        """ get the center position of the obstacle according the vertexes """
        center_x = 0.0
        center_y = 0.0

        for i in range(self.ver_num):
            center_x = center_x + self.vertexes[i, 0]
            center_y = center_y + self.vertexes[i, 1]

        center_x = center_x / self.ver_num
        center_y = center_y / self.ver_num

        # shape in (2, )
        self.init_position = np.array([center_x, center_y])
        self.position = np.copy(self.init_position)

    def get_vertex_vector(self):
        """ get the vertex vetcors """
        self.vertex_vertor = np.ones_like(self.vertexes)

        for i in range(self.ver_num):
            self.vertex_vertor[i, 0] = self.vertexes[i, 0] - self.position[0]
            self.vertex_vertor[i, 1] = self.vertexes[i, 1] - self.position[1]

    def get_edges(self):
        """ get the edges of the polytopic-shaped obstacle """
        self.edge_list = []
        for i in range(self.ver_num):
            edge = [
                self.vertexes[i, 0],
                self.vertexes[i, 1],
                self.vertexes[(i + 1) % self.ver_num, 0],
                self.vertexes[(i + 1) % self.ver_num, 1],
            ]
            self.edge_list.append(edge)

    def get_matrix(self):
        """ get Ax <= b """
        self.A = np.zeros((self.ver_num, 2))
        self.b = np.zeros((self.ver_num, 1))
   
        for i in range(self.ver_num):
            if i + 1 < self.ver_num:
                pre_point = self.vertexes[i]
                next_point = self.vertexes[i + 1]
            else:
                pre_point = self.vertexes[i]
                next_point = self.vertexes[0]

            diff = next_point - pre_point

            a = diff[1]
            b = -diff[0]
            c = a * pre_point[0] + b * pre_point[1]

            self.A[i, 0] = a
            self.A[i, 1] = b
            self.b[i, 0] = c

        return self.A, self.b

    def inside_obstacle(self, point):
        """ determine if a point is within an obstacle, boundary is not included"""
        assert point.shape == (2, 1)
        temp = self.A @ point - self.b
        return (self.A @ point < self.b).all(), temp

    def update_vertexes(self):
        """ update the vertexes when the obstacle moves, assume the obstacle only has translation """
        for i in range(self.ver_num):
            self.vertexes[i, 0] = self.position[0] + self.vertex_vertor[i, 0]
            self.vertexes[i, 1] = self.position[1] + self.vertex_vertor[i, 1]

    def get_current_vertexes(self, cur_position):
        """ get the vertexes according the cur position, assume the obstacle only has translation """
        cur_vertexes = np.ones_like(self.vertex_vertor)
        for i in range(self.ver_num):
            cur_vertexes[i, 0] = cur_position[0] + self.vertex_vertor[i, 0]
            cur_vertexes[i, 1] = cur_position[1] + self.vertex_vertor[i, 1]

        return cur_vertexes

    def arrive_destination(self):
        """ determine if the obstacle arrives its goal position """
        dist = np.linalg.norm(self.position.reshape(2, 1) - self.goal[0:2])

        if dist < 0.1:
            self.arrive_flag = True
            self.vel = np.zeros((2, ))
            return True
        else:
            self.arrive_flag = False
            return False

    def move_forward(self, step_time):
        """ move this obstacle """
        if self.mode != 'static':
            if self.arrive_flag:
                return
    
            self.position = self.position + self.vel * step_time
            if not self.arrive_flag:
                self.arrive_destination()

            self.update_vertexes()
            self.get_edges()
            self.get_matrix()

    def get_sampled_points(self, num_samples):
        """
        get the sample points from the obstacle's vertexes
        Params:
            obstacle vertexes: obstacle vertex posistion in anticlockwise (n, 2)
            num_samples: the points sampled in each edge

        Returns:
            sampled points in np.array (n, 2)
        """
        sample_points = []
        t = np.linspace(0, 1, num_samples)

        # sample points
        # TODO different edge lengths with different number of sampling points
        for i in range(self.ver_num):
            dx, dy = np.subtract(self.vertexes[(i + 1) % self.ver_num], self.vertexes[i]) 
            edge_points = [[self.vertexes[i][0] + dx * tt, self.vertexes[i][1] + dy * tt] for tt in t[:-1]]
            sample_points.extend(edge_points)

        return np.array(sample_points)
    
    def get_current_state(self):
        """ return the obstacle's position and velocity """
        current_state = np.array([self.position[0], self.position[1], self.vel[0], self.vel[1]])
        return current_state

if __name__ == '__main__':
    file_name = 'settings.yaml'
    with open(file_name) as file:
        config = yaml.safe_load(file)

    robot_params = config['robot']
    obs_params = config['circle_obstacle_list']
    # init