# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
from torch.nn import Module
from torch.optim import Optimizer
from gymnasium import Env
from src.data.services.memory import ReplayMemory
from src.data.models.transition import Transition
import torch
import numpy as np
from torch.nn import functional as F
from src.policy.services.epsilon_greedy import epsilon_greedy
from torch import device
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from src.pretrain.services.train import pre_train


def create_loss_graph(loss_list, graph_path):
    plt.plot(np.arange(len(loss_list)), loss_list, c="red")
    plt.title("Loss graph for DQN")
    plt.savefig(graph_path)


def train_step(
    env: Env,
    memory: ReplayMemory,
    dqn: Module,
    target_net: Module,
    optim: Optimizer,
    batch_size: int,
    gamma: float,
    device: str | device,
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
    non_final_next_states = (
        torch.stack(non_final_next_states)
        .squeeze()
        .reshape(len(non_final_next_states), -1)
        .to(device)
    )

    states = torch.stack(batch.state).squeeze().reshape(batch_size, -1).to(device)
    actions = torch.tensor(batch.action, device=device)
    rewards = torch.tensor(batch.reward, device=device)

    # predicted Q values
    state_action_values = dqn(states).gather(1, actions.unsqueeze(0))

    # next state discounted Q values
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_states_indices] = (
            target_net(non_final_next_states).max(1).values
        )

    # target Q values
    target_values = rewards + (gamma * next_state_values)

    # loss
    l = F.smooth_l1_loss(state_action_values, target_values.unsqueeze(0))

    # update
    optim.zero_grad()
    l.backward()
    torch.nn.utils.clip_grad_norm_(dqn.parameters(), 100)
    optim.step()

    # Return loss
    return l.item()


def train(
    env: Env,
    memory: ReplayMemory,
    dqn: Module,
    target_net: Module,
    optim: Optimizer,
    batch_size: int,
    gamma: float,
    device: str | device,
    num_episodes: int,
    epsilon: float,
    target_update: int,
    graph_path: str,
    allow_small_memory: bool = False,
    after_step_callback=None,
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
    @param graph_path: The complete path (from where the code is run) for where to
        store the graph that plots the loss. The path must be complete and include the suffix.
    @param allow_small_memory: If True, allow sampling from memory
        with less than batch_size transitions. The number sampled
        will be equal to the length of memory.
    @param after_step_callback: The callback to call after each step
    """
    losses = []
    called = set()
    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
        done = False
        while not done:
            action = epsilon_greedy(state, dqn, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    next_state, device=device, dtype=torch.float32
                ).unsqueeze(0)

            memory.push(state, action, next_state, reward)

            state = next_state

            new_loss = train_step(
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
            losses.append(new_loss)
            if after_step_callback is not None and episode not in called:
                after_step_callback(episode, dqn, device)
                called.add(episode)

        if episode % target_update == 0:
            target_net.load_state_dict(dqn.state_dict())

    # Create the loss graph
    create_loss_graph(losses, graph_path)


# deprecated
def full_training(
    pretrain_data_path: str,
    pretrain_epochs: int,
    pretrain_batch_size: int,
    pretrain_optimizer: Optimizer,
    env: Env,
    memory: ReplayMemory,
    dqn: Module,
    target_net: Module,
    optim: Optimizer,
    batch_size: int,
    gamma: float,
    device: str | device,
    num_episodes: int,
    epsilon: float,
    target_update: int,
    allow_small_memory: bool = False,
    pretrain_done_callback=None,
    pretrain_memory: ReplayMemory = None,
    after_step_callback=None,
):
    """
    Incorporates pretraining and normal training into one function

    It first pretrains the given dqn model before training it
    further on the given environment

    @param pretrain_data_path: The path to the pretraining data
    @param pretrain_epochs: The number of pretraining epochs
    @param pretrain_batch_size: The pretraining batch size
    @param pretrain_optimizer: The pretraining optimizer
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
    @param pretrain_done_callback: The callback to call after pretraining
    @param pretrain_samples: Replay memory sampled from dqn after pretraining
        has been performed, its samples will be added to given memory
    @param after_step_callback: The callback to call after each step
    """
    if pretrain_epochs > 0:
        pre_train(
            dqn,
            pretrain_optimizer,
            pretrain_data_path,
            pretrain_epochs,
            pretrain_batch_size,
            device=device,
        )
    target_net.load_state_dict(dqn.state_dict())
    if pretrain_done_callback is not None:
        pretrain_done_callback(dqn)
    if pretrain_memory is not None:
        memory.extend(pretrain_memory)
    train(
        env,
        memory,
        dqn,
        target_net,
        optim,
        batch_size,
        gamma,
        device,
        num_episodes,
        epsilon,
        target_update,
        allow_small_memory,
        after_step_callback=after_step_callback,
        graph_path="loss_graph.png",
    )
