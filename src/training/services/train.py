# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
from torch.nn import Module
from torch.optim import Optimizer
from gymnasium import Env
from src.data.services.memory import ReplayMemory, Transition


def train_step(
    env: Env,
    memory: ReplayMemory,
    dqn: Module,
    target_net: Module,
    optim: Optimizer,
    batch_size: int,
    gamma: float,
):
    """
    Given memory, sample batch_size number of transitions,
    extracting
    """
