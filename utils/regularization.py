import torch

def l1_regularization(model, lambda_l1):
    l1_reg = 0
    for param in model.parameters():
        l1_reg += torch.norm(param, 1)
    return lambda_l1 * l1_reg

def l2_regularization(model, lambda_l2):
    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param, 2)
    return lambda_l2 * l2_reg

def elastic_net_regularization(grad_params, lambda_l1, lambda_l2):
    erastic_reg = 0
    for param in grad_params.parameters():
        l1_reg = torch.norm(param, 1)
        l2_reg = torch.norm(param, 2)
        erastic_reg += lambda_l1 * l1_reg + lambda_l2 * l2_reg
    return erastic_reg