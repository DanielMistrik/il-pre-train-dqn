import torch
from src.policy.services.epsilon_greedy import epsilon_greedy as eg
from src.nn.dqn import DQN
from unittest.mock import patch


def test_epsilon_greedy_use_dqn():
    state_dim = 4
    action_dim = 2
    dqn = DQN(state_dim, action_dim, num_layers=1)
    state = torch.rand(1, state_dim)
    epsilon = 0
    action = eg(state, dqn, epsilon)
    expected = torch.argmax(dqn(state)).item()
    assert isinstance(action, int)
    assert action == expected


@patch("torch.randint")
def test_epsilon_greedy_use_random(mock_torch_randint):
    state_dim = 4
    action_dim = 2
    dqn = DQN(state_dim, action_dim, num_layers=1)
    state = torch.rand(1, state_dim)
    epsilon = 1
    mock_torch_randint.return_value = torch.tensor(-1)  # sentinel value
    action = eg(state, dqn, epsilon)
    expected = mock_torch_randint.return_value.item()
    assert isinstance(action, int)
    assert action == expected
