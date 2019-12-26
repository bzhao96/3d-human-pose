import torch
import torch.nn as nn
from torch.autograd import grad

LAMBDA = 10 # Gradient penalty lambda hyperparameter

def camera_loss(y_pred, device):
    k = y_pred.view(-1, 3, 2)
    k_m = torch.matmul(k.transpose(1, 2), k)
    inv_trace = 2 / (k_m[:, 0, 0] + k_m[:, 1, 1])
    loss = inv_trace.view(-1, 1, 1) * k_m - torch.eye(2).view(1,2,2).to(device)
    loss = torch.norm(loss, dim=(1,2))
    return loss

def gradient_penalty_loss(netD, real_data, fake_data, device):
    alpha = torch.rand(real_data.shape[0], 1).to(device)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)
    disc_interpolates = netD(interpolates)
    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                     grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                     create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


if __name__ == '__main__':
    a = torch.arange(12, dtype=torch.float).reshape(2,6)
    loss = camera_loss(a)
    print(loss)
