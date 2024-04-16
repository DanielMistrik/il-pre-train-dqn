import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    DQN takes in a vector representing the state/observation,
    and outputs a vector that represents the predicted Q values
    for each action conditioned on the input state.
    """

    def __init__(
        self, obs_dim, act_dim, num_layers=2, hidden_size=None, activation=F.relu
    ):
        """
        @param obs_dim: int, dimension of the input state/observation
        @param act_dim: int, dimension of the output action
        @param num_layers: int, number of hidden layers
        @param hidden_size: int, size of the hidden layers
            if not specified, it will be `obs_dim * 10`
        @param activation: function, activation function for hidden layers
        @return Tensor, output of the network

        Note the actual number of layers is `num_layers + 2` for the input and output layers
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size if hidden_size is not None else obs_dim * 10
        self.activation = activation

        self.first_layer = nn.Linear(self.obs_dim, self.hidden_size)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(self.hidden_size, self.act_dim)

    def forward(self, x):
        """
        Forward pass of the network
        @param x: Tensor, input to the network
        @return Tensor, output of the network
        """
        x = self.activation(self.first_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)
