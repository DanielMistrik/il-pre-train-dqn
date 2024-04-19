# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
from torch.nn import Module
from torch.optim import Optimizer
from gymnasium import Env
from src.data.services.memory import ReplayMemory
from src.data.models.transition import Transition
import torch
from torch.nn import functional as F
from src.policy.services.epsilon_greedy import epsilon_greedy


def train_step(
    env: Env,
    memory: ReplayMemory,
    dqn: Module,
    target_net: Module,
    optim: Optimizer,
    batch_size: int,
    gamma: float,
    device: str,
    allow_small_memory: bool = False,
):
    """
    Given memory, sample batch_size number of transitions,
    extracting

    @param env: The environment to train on
    @param memory: The replay memory to sample from
    @param dqn: The model to train
    @param target_net: network used to help calculate target Q values
    @param optim: The optimizer to use
    @param batch_size: The number of transitions to sample
    @param gamma: The discount factor
    @param device: The device to use for training
    @param allow_small_memory: If True, allow sampling from memory
        with less than batch_size transitions. The number sampled
        will be equal to the length of memory.
    """
    if len(memory) < batch_size and not allow_small_memory:
        return
    samples = memory.sample(min(batch_size, len(memory)))
    batch = Transition(*zip(*samples))

    # when calculating the target Q values, states that are the terminal
    # states will by definition have value of 0
    non_final_states_indices = [x is not None for x in batch.next_state]
    non_final_states_indices = torch.tensor(
        non_final_states_indices, dtype=torch.bool, device=device
    )
    non_final_next_states = [x for x in batch.next_state if x is not None]
    non_final_next_states = torch.stack(non_final_next_states).to(device)

    states = torch.stack(batch.state).to(device)
    actions = torch.tensor(batch.action, device=device)
    rewards = torch.tensor(batch.reward, device=device)

    # predicted Q values
    state_action_values = dqn(states).gather(1, actions.unsqueeze(-1))

    # next state discounted Q values
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_states_indices] = (
            target_net(non_final_next_states).max(1).values
        )

    # target Q values
    target_values = rewards + (gamma * next_state_values)

    # loss
    l = F.smooth_l1_loss(state_action_values, target_values.unsqueeze(1))

    # update
    optim.zero_grad()
    l.backward()
    torch.nn.utils.clip_grad_norm_(dqn.parameters(), 100)
    optim.step()


def train(
    env: Env,
    memory: ReplayMemory,
    dqn: Module,
    target_net: Module,
    optim: Optimizer,
    batch_size: int,
    gamma: float,
    device: str,
    num_episodes: int,
    epsilon: float,
    target_update: int,
    allow_small_memory: bool = False,
):
    """
    Train a DQN model using the given environment.

    @param env: The environment to train on
    @param memory: The replay memory to sample from
    @param dqn: The model to train
    @param target_net: network used to help calculate target Q values
    @param optim: The optimizer to use
    @param batch_size: The number of transitions to sample
    @param gamma: The discount factor
    @param device: The device to use for training
    @param num_episodes: The number of episodes to train for
    @param epsilon: The probability of taking a random action
    @param target_update: The number of episodes to update the target network
    @param allow_small_memory: If True, allow sampling from memory
        with less than batch_size transitions. The number sampled
        will be equal to the length of memory.
    """
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            state = torch.tensor(state, device=device).unsqueeze(0)
            action = epsilon_greedy(state, dqn, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(next_state, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward)

            state = next_state

            train_step(
                env,
                memory,
                dqn,
                target_net,
                optim,
                batch_size,
                gamma,
                device,
                allow_small_memory,
            )

        if episode % target_update == 0:
            target_net.load_state_dict(dqn.state_dict())
