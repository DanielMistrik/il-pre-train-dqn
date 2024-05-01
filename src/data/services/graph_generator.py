import matplotlib.pyplot as plt
import numpy as np

def gen_graph(data):
    """
    Will generate a graph given the data.
    :param data: Assumes the following format:
    {
        "legend_name":[(value_at_100_episodes, sd_at_100_episodes),...,
                        (value_at_4999_episodes, sd_at_4999_episodes)]
    }
    len(data) <= 6
    """
    plt.figure(figsize=(10, 6))

    colors = ['black', 'blue', 'red', 'green', 'yellow', 'purple']
    episode_values = [100, 500, 1000, 2000, 3000, 4999]
    # Plot with error bars and filled error areas
    i = 0
    for key in data.keys():
        y_vals = np.array([value[0] for value in data[key]])
        std_vals = np.array([value[1] for value in data[key]])
        plt.errorbar(episode_values, y_vals, label=str(key), color=colors[i], alpha=0.5)
        plt.fill_between(episode_values, y_vals - std_vals, y_vals + std_vals, color=colors[i], alpha=0.2)
        i += 1

    # Add legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, ncol=2, fontsize='small')

    # Add labels
    #plt.xscale('log')
    plt.xticks(episode_values, episode_values)
    plt.xlabel('Episodes Trained ')
    plt.ylabel('Performance')

    # Show plot
    plt.show()


if __name__ == '__main__':
    mc_data = {
        'Vanilla DQN (No Pretraining; Target Update Every 1000 Epochs)': [(-200.0, 0.0), (-200.0, 0.0), (-200.0,0.0), (-200.0, 0.0), (-200.0,0.0), (-200.0,0.0)],
        'DQN (10 Epochs Pretraining; Target Update Every 1000 Epochs)': [(-155.3,41.71), (-114.8, 23.61), (-108.3, 0.46), (-133.1, 22.3), (-128.1, 32.52), (-113.7, 2.69)],
        'DQN (100 Epochs Pretraining; Target Update Every 1000 Epochs)': [(-155.0, 38.51), (-137.4 ,42.97), (-105.3, 5.12), (-129.2, 28.75), (-127.8, 32.84), (-200.0, 0)],
        'DQN (500 Epochs Pretraining; Target Update Every 1000 Epochs)': [(-175.4, 33.54), (-188.8, 17.11), (-199.8, 0.4), (-200.0, 0.0), (-200.0, 0.0), (-200.0,0.0)],
    }
    ## Need to fill these in
    ll_data = {
        'Vanilla DQN (No Pretraining; Target Update Every 1000 Epochs)': [(-200.0, 0.0), (-200.0, 0.0), (-200.0, 0.0),
                                                                          (-200.0, 0.0), (-200.0, 0.0), (-200.0, 0.0)],
        'DQN (10 Epochs Pretraining; Target Update Every 1000 Epochs)': [(-155.3, 41.71), (-114.8, 23.61),
                                                                         (-108.3, 0.46), (-133.1, 22.3),
                                                                         (-128.1, 32.52), (-113.7, 2.69)],
        'DQN (100 Epochs Pretraining; Target Update Every 1000 Epochs)': [(-155.0, 38.51), (-137.4, 42.97),
                                                                          (-105.3, 5.12), (-129.2, 28.75),
                                                                          (-127.8, 32.84), (-200.0, 0)],
        'DQN (500 Epochs Pretraining; Target Update Every 1000 Epochs)': [(-175.4, 33.54), (-188.8, 17.11),
                                                                          (-199.8, 0.4), (-200.0, 0.0), (-200.0, 0.0),
                                                                          (-200.0, 0.0)],
    }
    ab_data = {
        'Vanilla DQN (No Pretraining; Target Update Every 1000 Epochs)': [(-200.0, 0.0), (-200.0, 0.0), (-200.0, 0.0),
                                                                          (-200.0, 0.0), (-200.0, 0.0), (-200.0, 0.0)],
        'DQN (10 Epochs Pretraining; Target Update Every 1000 Epochs)': [(-155.3, 41.71), (-114.8, 23.61),
                                                                         (-108.3, 0.46), (-133.1, 22.3),
                                                                         (-128.1, 32.52), (-113.7, 2.69)],
        'DQN (100 Epochs Pretraining; Target Update Every 1000 Epochs)': [(-155.0, 38.51), (-137.4, 42.97),
                                                                          (-105.3, 5.12), (-129.2, 28.75),
                                                                          (-127.8, 32.84), (-200.0, 0)],
        'DQN (500 Epochs Pretraining; Target Update Every 1000 Epochs)': [(-175.4, 33.54), (-188.8, 17.11),
                                                                          (-199.8, 0.4), (-200.0, 0.0), (-200.0, 0.0),
                                                                          (-200.0, 0.0)],
    }
    gen_graph(mc_data)