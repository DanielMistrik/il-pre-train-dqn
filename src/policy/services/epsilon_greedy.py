import torch


def epsilon_greedy(state, dqn, epsilon):
    """
    Epsilon-greedy policy takes a random action with probability epsilon,
    otherwise takes the best action according to the Q-network.

    @param state: Tensor, current state of the environment
        with shape (state_dim,) or (1, state_dim)
    @param dqn: DQN, the Q-network
    @param epsilon: float, probability of taking a random action
    @return int, action to take
    """
    if torch.rand(1).item() < epsilon:
        return torch.randint(0, dqn.act_dim, (1,)).item()
    else:
        with torch.no_grad():
            return torch.argmax(dqn(state)).item()
