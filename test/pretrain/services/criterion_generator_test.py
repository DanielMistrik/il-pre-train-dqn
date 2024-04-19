import torch
from src.pretrain.services.criterion_generator import CEP_Loss


test_CEP_criterion = CEP_Loss()

def test_cep_Loss_single_value_no_weights():
    q_function_vals = torch.tensor([-1.2, 3.4, 2.2])
    target_action = 2 #0-indexed
    # -(log(P(2>0)) + log(P(2>1))) = -(log(e^(2.2)/(e^(2.2)+e^(-1.2)))+log(e^(2.2)/(e^(2.2)+e^(3.4)))=-(-1.4961)=1.4961
    expected_cep_loss = torch.tensor(1.4961)
    real_cep_loss = test_CEP_criterion(q_function_vals, target_action)
    assert torch.isclose(expected_cep_loss, real_cep_loss).item()


def test_cep_Loss_single_value_with_weights():
    weights = torch.tensor([1, 4, 0.5])
    q_function_vals = torch.tensor([-1.2, 3.4, 2.2])
    target_action = 0  # 0-indexed
    # -(log(P(2>0)) + log(P(2>1))) = -(0.5*log(e^(-1.2)/(e^(2.2)+e^(-1.2)))+4*log(e^(-1.2)/(e^(-1.2)+e^(3.4)))=20.156
    expected_cep_loss = torch.tensor(20.156)
    real_cep_loss = test_CEP_criterion(q_function_vals, target_action, weights)
    assert torch.isclose(expected_cep_loss, real_cep_loss, atol=1e-3).item()