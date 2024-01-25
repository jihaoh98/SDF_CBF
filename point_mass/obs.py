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
    


