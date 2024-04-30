import os
import json
import pickle

def _get_nums(strng):
    """
    Extracts all the numbers in the string strng into an ordered list of integers.
    Assumes all numbers are positive
    """
    int_list = []
    curr_int = 0
    hit_int = False
    for char in strng:
        if char.isnumeric():
            curr_int = curr_int * 10 + int(char)
            hit_int = True
        elif hit_int:
            int_list.append(curr_int)
            curr_int = 0
            hit_int = False

    return int_list


def get_result_stats(env_name, num_episodes, num_epochs, num_target_update):
    """
    Returns the average and variance of the given result with the episode length,
    pre-training epochs and num_target_variables for the given model env_name
    """
    total_reward_list = []
    for file in os.listdir(f'../../../tmp/{env_name}/'):
        int_list = _get_nums(file)
        # One of the files we are interested in
        if ('after_training' in file and int_list[1] == num_episodes and int_list[3] == num_epochs and\
                int_list[4] == num_target_update):
            with open(f'../../../tmp/{env_name}/{file}', "r") as read_file:
                data = json.load(read_file)
                total_reward_list.append(sum(data))

    n = len(total_reward_list)
    mean_val = sum(total_reward_list) / n
    st_dev = (sum([(val - mean_val)**2 for val in total_reward_list]) / n)**0.5
    return mean_val, st_dev


def collect_data_on_model(env_name):
    """
    Returns a dictionary with mean reward and standard dev for all the configurations of the DQN trained
    for the env_name environment
    """
    return_dict = {}
    # Episodes range in 100,500,1000,2000,3000,4999
    for ep in [100, 500, 1000, 2000, 3000, 4999]:
        # Pre train epochs range in 10, 100, 500
        for pt_epch in [0, 10, 100, 500]:
            # Target Update range i 100, 500, 1000
            for trgt_update in [100, 500, 1000]:
                key = f'{env_name}_trained with {ep} episodes, pre-trained on {pt_epch} epochs with target update of {trgt_update}'
                return_dict[key] = get_result_stats(env_name, ep, pt_epch, trgt_update)
    return return_dict

if __name__ == "__main__":
    #print(get_result_stats('LunarLander-v2', 100, 100, 100))
    print(collect_data_on_model('MountainCar-v0'))