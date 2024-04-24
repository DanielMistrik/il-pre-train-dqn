import numpy as np
import gymnasium as gym
from gymnasium.utils.play import play
from data_collect import env_names_to_metadata


episode_rewards = []
current_reward = 0


def reward_collect_callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    global current_reward
    current_reward += rew
    if terminated or truncated:
        print(rew)
        episode_rewards.append(current_reward)
        current_reward = 0


def data_collect(num_samples: int, data_env_name: str) -> None:
    """
    Initializes the provided environment and collects num_samples samples and saves it into the data directory.
    Assumes its run from directory this file is in, if not provide the relative path to this directory (with an ending
    backslash).
    """
    # Initializaiton
    global episode_rewards
    global current_reward

    episode_rewards = []
    data_env = gym.make(data_env_name, render_mode="rgb_array")
    # Collect rewards from human player
    while len(episode_rewards) < num_samples:
        current_reward = 0
        play(data_env, callback=reward_collect_callback, keys_to_action=env_names_to_metadata[data_env_name][0],
             noop=env_names_to_metadata[data_env_name][1], fps=env_names_to_metadata[data_env_name][2])
    # Initial and final observations might be 'tainted' so we skip them
    number_of_full_episodes = len(episode_rewards)
    front_padding = (number_of_full_episodes - num_samples)//2
    end_padding = len(episode_rewards) - front_padding
    np_collected_data = np.array(episode_rewards[front_padding:end_padding], dtype=object)
    return np_collected_data, np.mean(np_collected_data)


if __name__ == '__main__':
    print(data_collect(10, 'LunarLander-v2'))