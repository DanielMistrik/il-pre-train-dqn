# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
from collections import deque
import random
from src.data.models.transition import Transition


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """
        Expected order of arguments:
        state, action, next_state, reward

        where state and next_state are torch tensors
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def get_memory(self):
        return self.memory

    def extend(self, memory):
        """
        Extends the current memory with the memory passed as argument
        """
        self.memory += memory.get_memory()
