import gymnasium as gym
import torch
from src.data.services.dimensions import get_dims
import json
from torch.optim import Adam
from torch.optim import Optimizer
from src.nn.services.dqn import DQN
from src.training.services.train import train
from src.pretrain.services.train import pre_train
from src.data.services.memory import ReplayMemory
import cv2
from src.policy.services.epsilon_greedy import epsilon_greedy
from tqdm.auto import tqdm
from typing import List
import random
from typing import Optional


def create_video(frames, fps=15, output_name="output"):
    out = cv2.VideoWriter(
        f"{output_name}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frames[0].shape[1], frames[0].shape[0]),
    )
    for frame in frames:
        out.write(frame)
    out.release()


def test_dqn(dqn, env, device, output_name="output", output_dir="./tmpvideo"):
    if isinstance(env, str):
        env = gym.make(env, render_mode="rgb_array")
    frames = []
    rewards = []
    state, _ = env.reset()
    done = False
    while not done:
        frames.append(env.render())
        action = epsilon_greedy(
            torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device), dqn, 0
        )
        next_state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
        state = next_state
    if output_dir is not None:
        create_video(frames, output_name=f"{output_dir}/{output_name}")
    return rewards


def easy_train(
    env_name: str,
    pretrain_data_path: str,
    batch_size: int = 64,
    gamma: float = 0.99,
    num_training_episodes: int = 5000,
    target_update: int = 100,
    epsilon: float = 0.2,
    train_replay_memory_size: int = 1000000,
    train_optm_class: Optimizer = Adam,
    train_lr: float = 1e-6,
    pretrain_batch_size: int = 64,
    pretrain_epochs: int = 1000,
    pretrain_optm_class: Optimizer = Adam,
    pretrain_lr: float = 0.001,
    pretrain_replay_samples: int = 1000,
    train_vis_points: List[int] = [],
    train_vis_trials: int = 10,
    train_vis_trial_saves: int = 2,
    device: str | torch.device = "cpu",
    video_output_dir: str = "./tmpvideo",
    rewards_output_dir: str = "./tmp",
    graph_output_dir: str = "./tmpgraphs",
    save_model_path: Optional[str] = "./tmpmodel",
) -> DQN:
    def after_test_callback(episode, dqn, device):
        if episode in train_vis_points:
            with torch.no_grad():
                save_indices = random.sample(
                    range(train_vis_trials), train_vis_trial_saves
                )
                for i in range(train_vis_trials):
                    name = f"after_training_{env_name}_{episode}_sample_{i}"
                    rewards = test_dqn(
                        dqn,
                        gym.make(env_name, render_mode="rgb_array"),
                        device,
                        output_name=name,
                        output_dir=video_output_dir if i in save_indices else None,
                    )
                    with open(f"{rewards_output_dir}/{name}.json", "w") as f:
                        json.dump(rewards, f)
                    return rewards

    env = gym.make(env_name)
    state_dim, action_dim = get_dims(env)
    dqn = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    before_training_rewards = test_dqn(
        dqn,
        env_name,
        device,
        output_name=f"before_training_{env_name}",
        output_dir=video_output_dir,
    )
    with open(f"{rewards_output_dir}/before_training_{env_name}.json", "w") as f:
        json.dump(before_training_rewards, f)
    pretrain_optm = pretrain_optm_class(dqn.parameters(), lr=pretrain_lr)
    pre_train(
        dqn,
        pretrain_optm,
        pretrain_data_path,
        pretrain_epochs,
        pretrain_batch_size,
        device=device,
    )
    pretrained_rewards = test_dqn(
        dqn,
        env_name,
        device,
        output_name=f"pretrained_{env_name}",
        output_dir=video_output_dir,
    )
    with open(f"{rewards_output_dir}/pretrained_{env_name}.json", "w") as f:
        json.dump(pretrained_rewards, f)
    pretrain_memory = ReplayMemory(pretrain_replay_samples)
    with torch.no_grad():
        for _ in tqdm(range(pretrain_replay_samples)):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            done = False
            while not done:
                action = epsilon_greedy(state, dqn, 0)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                if terminated:
                    next_state = None
                else:
                    next_state = (
                        torch.tensor(next_state, dtype=torch.float32)
                        .unsqueeze(0)
                        .to(device)
                    )
                pretrain_memory.push(state, action, next_state, reward)
                state = next_state
    target_net.load_state_dict(dqn.state_dict())
    optm = train_optm_class(dqn.parameters(), lr=train_lr)
    mem = ReplayMemory(train_replay_memory_size)
    train(
        env=env,
        memory=mem,
        dqn=dqn,
        target_net=target_net,
        optim=optm,
        batch_size=batch_size,
        gamma=gamma,
        device=device,
        num_episodes=num_training_episodes,
        epsilon=epsilon,
        target_update=target_update,
        graph_path=f"{graph_output_dir}/{env_name}_{num_training_episodes}_episodes.png",
        after_step_callback=after_test_callback,
    )
    if save_model_path is not None:
        torch.save(
            {
                "env_name": env_name,
                "dqn": dqn.state_dict(),
                "kwargs": dqn.kwargs,
                "trained_episodes": num_training_episodes,
            },
            save_model_path,
        )
    return dqn
