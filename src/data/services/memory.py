# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
from collections import deque
import random
from src.data.models.transition import Transition


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
