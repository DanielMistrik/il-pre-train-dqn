import torch
from src.pretrain.services.criterion_generator import CEP_Loss, Classification_Loss


test_CEP_criterion = CEP_Loss()
test_class_criterion = Classification_Loss()


def test_cep_loss_single_value_no_weights():
    q_function_vals = torch.tensor([[-1.2, 3.4, 2.2]])
    target_action = torch.tensor([2]) #0-indexed
    # -(log(P(2>0)) + log(P(2>1))) = -(log(e^(2.2)/(e^(2.2)+e^(-1.2)))+log(e^(2.2)/(e^(2.2)+e^(3.4)))=-(-1.4961)=1.4961
    expected_cep_loss = torch.tensor(1.4961)
    real_cep_loss = test_CEP_criterion(q_function_vals, target_action)
    assert torch.isclose(expected_cep_loss, real_cep_loss).item()


def test_cep_loss_batch_no_weights():
    q_function_vals = torch.tensor([[-1.2, 3.4, 2.2], [-1.2, 3.4, 2.2]])
    target_action = torch.tensor([2, 2]) #0-indexed
    # The above times 2
    expected_cep_loss = torch.tensor(2.9922)
    real_cep_loss = test_CEP_criterion(q_function_vals, target_action)
    assert torch.isclose(expected_cep_loss, real_cep_loss).item()


def test_cep_loss_single_value_with_weights():
    weights = torch.tensor([[1, 4, 0.5]])
    q_function_vals = torch.tensor([[-1.2, 3.4, 2.2]])
    target_action = torch.tensor([0])  # 0-indexed
    # -(log(P(2>0)) + log(P(2>1))) = -(0.5*log(e^(-1.2)/(e^(2.2)+e^(-1.2)))+4*log(e^(-1.2)/(e^(-1.2)+e^(3.4)))=20.156
    expected_cep_loss = torch.tensor(20.156)
    real_cep_loss = test_CEP_criterion(q_function_vals, target_action, weights)
    assert torch.isclose(expected_cep_loss, real_cep_loss, atol=1e-3).item()


def test_cep_loss_batch_value_with_weights():
    weights = torch.tensor([[1, 4, 0.5], [1, 4, 0.5]])
    q_function_vals = torch.tensor([[-1.2, 3.4, 2.2], [-1.2, 3.4, 2.2]])
    target_action = torch.tensor([0, 0])  # 0-indexed
    # Use calculations from above (times 2)
    expected_cep_loss = torch.tensor(40.312)
    real_cep_loss = test_CEP_criterion(q_function_vals, target_action, weights)
    print(torch.isclose(expected_cep_loss, real_cep_loss, atol=1e-3))
    assert torch.all(torch.isclose(expected_cep_loss, real_cep_loss, atol=1e-3))


def test_class_loss_single_value():
    q_function_vals = torch.tensor([[-1.2, 3.4, 2.2]])
    bad_action = torch.tensor([2])
    assert 1 == test_class_criterion(q_function_vals, bad_action).item()
    good_action = torch.tensor([1])
    assert 0 == test_class_criterion(q_function_vals, good_action).item()


def test_class_loss_batch():
    q_function_vals = torch.tensor([[-1.2, 3.4, 2.2], [-1.2, 3.4, 2.2]])
    bad_actions = torch.tensor([2, 0])
    assert 2 == test_class_criterion(q_function_vals, bad_actions).item()
    good_actions = torch.tensor([1, 1])
    assert 0 == test_class_criterion(q_function_vals, good_actions).item()
    mixed_actions = torch.tensor([1, 0])
    assert 1 == test_class_criterion(q_function_vals, mixed_actions).item()

