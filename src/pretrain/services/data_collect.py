import gymnasium as gym
import numpy as np
import os
from gymnasium.utils.play import play

collected_data = []
# Each value is a tuple of: key_to_action dict, noop action, preffered fps
env_names_to_metadata = {
    'LunarLander-v2': (
        {
            'w': 2,
            'a': 1,
            'd': 3,
        }, 0, 30
    ),
    'Acrobot-v1': (
        {
            'a': 0,
            'd': 2,
        }, 1, 30
    ),
    'MountainCar-v0': (
        {
            'a': 0,
            'd': 2,
        }, 1, 30
    )
}


def data_collect_callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    if isinstance(obs_t, tuple):
        obs_t = obs_t[0]
    collected_data.append((obs_t, action))


def data_collect(num_samples: int, data_env_name: str, path_to_dir: str) -> None:
    """
    Initializes the provided environment and collects num_samples samples and saves it into the data directory.
    Assumes its run from directory this file is in, if not provide the relative path to this directory (with an ending
    backslash).
    """
    global collected_data
    collected_data = []
    data_env = gym.make(data_env_name, render_mode="rgb_array")
    while len(collected_data) < num_samples:
        play(data_env, callback=data_collect_callback, keys_to_action=env_names_to_metadata[data_env_name][0],
             noop=env_names_to_metadata[data_env_name][1], fps=env_names_to_metadata[data_env_name][2])
    np_collected_data = np.array(collected_data[:num_samples], dtype=object)
    # Find available name for new collected data
    prefix = '' if path_to_dir is None else path_to_dir
    available_sample_id = len([file for file in os.listdir(prefix + '../data/') if data_env_name in file])
    np.save(f"{prefix}../data/{data_env_name}_{available_sample_id}", np_collected_data)


if __name__ == '__main__':
    data_collect(10, 'Acrobot-v1', None)
