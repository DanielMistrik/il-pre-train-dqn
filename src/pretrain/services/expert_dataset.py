from torch.utils.data import Dataset
import numpy as np


class ExpertDataset(Dataset):
    def __init__(self, data_path):
        raw_numpy_input = np.load(data_path, allow_pickle=True)
        self.actions = [item[1] for item in raw_numpy_input]
        self.obs = [item[0] for item in raw_numpy_input]

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]


if __name__ == '__main__':
    dataset = ExpertDataset('../data/MountainCar-v0_10_000.npy')
    print(len(dataset))
    print(dataset.__getitem__(3))