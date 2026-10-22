import numpy as np
import torch
import torch.nn as nn

KS = [1, 1, 3, 3]
BETA = [1., .25, .5, .12]

class NeuralHawkes(nn.Module):
    def __init__(self, hidden=64):
        self.mu = nn.Parameter(torch.full((U,), -5.))
        self.A = nn.Parameter(torch.full((U, U, 4), -4.))
        self.beta = nn.Parameter(torch.tensor(BETA).log())

    def alpha(self, x):
        return F.softplus(self.enc(x)).view(*x.shape[:2], U, 4)


    def forward(self, x, ta, te, ue):
        a = self.alpha(x)
        dt = te[:, :, None] - ta[:, None, :]
        act = (self.g(dt) * a.gather(2, ue[:, None, :, None].expand(-1, x.shape[1], -1, 4)).permute(0, 2, 1, 3)).sum(-1)
        dt = te[:, :, None] - te[:, None, :]
        usr = (self.g(dt) * F.softplus(self.A)[ue[:, :, None], ue[:, None, :]]).sum(-1)
        bg = F.softplus(self.mu)[ue]
        return bg + act.sum(-1) + usr.sum(-1)

def fit(model, loss_fn, data, epochs=100, lr=.01):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model, data)
        loss.backward()
        opt.step()
    return model