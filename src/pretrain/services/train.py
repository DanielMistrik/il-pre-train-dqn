import tqdm
import torch
import numpy as np
from torch.optim import Adam
import matplotlib.pyplot as plt
from src.nn.services.dqn import DQN
from torch.utils.data import DataLoader
from expert_dataset import ExpertDataset
from criterion_generator import CEP_Loss


def eval_model(model, loss_fn, val_loader):
    """
    Returns Validation Loss based on val_loader data and loss_fn
    """
    total_loss = []
    with torch.no_grad():
        for sample in val_loader:
            pred_q_val = model.forward(sample[0])
            loss = loss_fn(pred_q_val, sample[1])
            total_loss.append(loss.item())

    return sum(total_loss)/len(total_loss)


def run_epoch(model, loss_fn, optimizer, train_loader, val_loader, train_loss, valid_loss):
    """
    Returns updated train,validation losses
    """
    batch_train_loss = []
    for sample in train_loader:
        # Generate Loss
        pred_q_val = model.forward(sample[0])
        loss = loss_fn(pred_q_val, sample[1])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_train_loss.append(loss.item())

    valid_loss.append(eval_model(model, loss_fn, val_loader))
    train_loss.append(sum(batch_train_loss)/len(batch_train_loss))

    return train_loss, valid_loss


def gen_tv_graph(train_loss, valid_loss, data_path):
    plt.plot(np.arange(len(train_loss)), train_loss, c='blue', label="Train")
    plt.plot(np.arange(len(valid_loss)), valid_loss, c='red', label="Validation")
    plt.title(f'Train-Validation Graph for model trained on dataset {data_path}')
    plt.legend()
    path_to_data_dir = data_path[:data_path.rfind('/')]
    plt.savefig(path_to_data_dir+"/tv_graph_"+data_path[data_path.rfind('/')+1:data_path.rfind('.npy')+1]+".png")


def pre_train(model, optimizer, data_path, num_epochs, batch_size=64):
    """
    Assumes data_path is path to the data directory within pretrain for a .npy file
    """
    # Set-Up
    expert_dataset = ExpertDataset(data_path)
    # 80-20 Split
    train_size, val_size = len(expert_dataset) - len(expert_dataset)//5, len(expert_dataset)//5
    train_dataset, val_dataset = torch.utils.data.random_split(expert_dataset, [train_size, val_size])
    train_loader = DataLoader(expert_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    loss_fn = CEP_Loss()
    train_loss, valid_loss = [], []
    # Train for epochs
    for _ in tqdm.tqdm(range(num_epochs)):
        train_loss, valid_loss = run_epoch(model, loss_fn, optimizer, train_loader, val_loader, train_loss, valid_loss)
    # Produce TV Loss Images
    gen_tv_graph(train_loss, valid_loss, data_path)


if __name__ == '__main__':
    # Test on MountainCar-v0
    dqn = DQN(2, 3)
    optm = Adam(dqn.parameters(), lr=0.001)
    pre_train(dqn, optm, '../data/MountainCar-v0_0.npy', 10, 32)

