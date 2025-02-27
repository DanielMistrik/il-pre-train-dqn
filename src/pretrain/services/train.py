import tqdm
import torch
import numpy as np
from torch.optim import Adam
import matplotlib.pyplot as plt
from src.nn.services.dqn import DQN
from torch.utils.data import DataLoader
from src.pretrain.services.expert_dataset import ExpertDataset
from src.pretrain.services.criterion_generator import CEP_Loss, Classification_Loss
from src.pretrain.services.data_collect import env_names_to_metadata


def eval_model(model, loss_fn, class_loss_fn, val_loader, device="cpu"):
    """
    Returns Validation Loss and Classification (1-0) loss based on val_loader data and loss_fn
    """
    total_loss = []
    total_class_loss = []
    with torch.no_grad():
        for sample in val_loader:
            pred_q_val = model.forward(sample[0].to(device))
            loss = loss_fn(pred_q_val, sample[1].to(device))
            closs = class_loss_fn(pred_q_val, sample[1].to(device))
            total_loss.append(loss.item())
            total_class_loss.append(closs.item())

    return sum(total_loss) / len(total_loss), sum(total_class_loss) / len(total_class_loss)


def run_epoch(
    model,
    loss_fn,
    class_loss_fn,
    optimizer,
    train_loader,
    val_loader,
    train_loss,
    valid_loss,
    class_loss,
    device="cpu",
):
    """
    Returns updated train,validation losses
    """
    batch_train_loss = []
    for sample in train_loader:
        # Generate Loss
        pred_q_val = model.forward(sample[0].to(device))
        loss = loss_fn(pred_q_val, sample[1].to(device))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_train_loss.append(loss.item())

    new_val_loss, new_class_loss = eval_model(model, loss_fn, class_loss_fn, val_loader, device=device)
    valid_loss.append(new_val_loss)
    class_loss.append(new_class_loss)
    train_loss.append(sum(batch_train_loss) / len(batch_train_loss))

    return train_loss, valid_loss, class_loss


def _get_model_name_from_data_path(data_path):
    for env_name in env_names_to_metadata.keys():
        if env_name in data_path:
            return env_name


def gen_loss_graphs(train_loss, valid_loss, clasif_error, data_path):
    plt.plot(np.arange(len(train_loss)), train_loss, c="blue", label="Train")
    plt.plot(np.arange(len(valid_loss)), valid_loss, c="red", label="Validation")
    plt.title(
        f"Train-Validation Graph for model trained on {_get_model_name_from_data_path(data_path)}"
    )
    plt.legend()
    path_to_data_dir = data_path[: data_path.rfind("/")]
    plt.savefig(
        path_to_data_dir
        + "/tv_graph_"
        + data_path[data_path.rfind("/") + 1: data_path.rfind(".npy") + 1]
        + ".png"
    )
    plt.close()
    plt.plot(np.arange(len(clasif_error)), clasif_error, c="green", label="Classification Error")
    plt.title(
        f"Classification Loss Graph for model trained on {_get_model_name_from_data_path(data_path)}"
    )
    plt.legend()
    path_to_data_dir = data_path[: data_path.rfind("/")]
    plt.savefig(
        path_to_data_dir
        + "/class_graph_"
        + data_path[data_path.rfind("/") + 1: data_path.rfind(".npy")]
        + ".png"
    )


def pre_train(model, optimizer, data_path, num_epochs, batch_size=64, device="cpu"):
    """
    Assumes data_path is path to the data directory within pretrain for a .npy file
    """
    # Set-Up
    expert_dataset = ExpertDataset(data_path)
    # 80-20 Split
    train_size, val_size = (
        len(expert_dataset) - len(expert_dataset) // 5,
        len(expert_dataset) // 5,
    )
    train_dataset, val_dataset = torch.utils.data.random_split(
        expert_dataset, [train_size, val_size]
    )
    train_loader = DataLoader(expert_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    loss_fn = CEP_Loss()
    class_loss_fn = Classification_Loss()
    train_loss, valid_loss, class_loss = [], [], []
    # Train for epochs
    for _ in tqdm.tqdm(range(num_epochs)):
        train_loss, valid_loss, class_loss = run_epoch(
            model,
            loss_fn,
            class_loss_fn,
            optimizer,
            train_loader,
            val_loader,
            train_loss,
            valid_loss,
            class_loss,
            device=device,
        )
    # Produce TV Loss Images
    gen_loss_graphs(train_loss, valid_loss, class_loss, data_path)


if __name__ == "__main__":
    # Test on LunarLander-v2
    dqn = DQN(8, 4)
    optm = Adam(dqn.parameters(), lr=0.001)
    pre_train(dqn, optm, "../data/LunarLander-v2_10_000.npy", 100, 16)
