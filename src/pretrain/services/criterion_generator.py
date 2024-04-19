# Generates loss functions for our environments
import torch
from torch import nn


def cum_p(q_function_a: torch.int, q_function_b: torch.int):
    """
    Bradley and Terry Model adapted with Q-Values - P(a>b|S=s, Q=q_function)
    Where you provide the q-function vals q(s,a) & q(s,b) respectively
    Our approximation is that the Q-value represents a rough (un-normalized) probability of choosing that action
    """
    return torch.div(
        torch.exp(q_function_a),
        torch.add(torch.exp(q_function_a), torch.exp(q_function_b))
    )


class CEP_Loss(nn.Module):
    def __init__(self):
        super(CEP_Loss, self).__init__()

    def forward(self, pred_q_vals: torch.Tensor, target_action: torch.int, weights: torch.Tensor=None):
        weights = torch.ones(pred_q_vals.shape[0]) if weights is None else weights
        weights[target_action] = 0 #We aren't interested in P(i>i) for our loss function
        target_q_val = pred_q_vals[target_action] * torch.ones(pred_q_vals.shape[0])
        return torch.mul(-1, torch.mul(weights, torch.log(cum_p(target_q_val, pred_q_vals))).sum())
