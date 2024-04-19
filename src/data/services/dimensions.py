from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from typing import Tuple


def get_dims(env: Env) -> Tuple[int, int]:
    """
    Given an environment, return the number of states and actions
    """
    state_type = env.observation_space
    action_type = env.action_space
    if not isinstance(action_type, Discrete):
        raise ValueError("Only discrete action spaces are supported")
    action_dim = action_type.n
    if isinstance(state_type, Box):
        state_dim = state_type.shape[0]
    elif isinstance(state_type, Discrete):
        state_dim = 1
    else:
        raise ValueError("Only Box and Discrete state spaces are supported")
    return state_dim, action_dim
