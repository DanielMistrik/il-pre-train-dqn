import torch
from src.policy.services.epsilon_greedy import epsilon_greedy as eg
from src.nn.dqn import DQN


def test_epsilon_greedy():
    state_dim = 4
    action_dim = 2
    dqn = DQN(state_dim, action_dim, num_layers=1)
    state = torch.rand(1, state_dim)
    epsilon = 0
    action = eg(state, dqn, epsilon)
    expected = torch.argmax(dqn(state)).item()
    assert isinstance(action, int)
    assert action == expected
