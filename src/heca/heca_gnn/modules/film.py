from torch import nn


class FiLM(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.generator = nn.Linear(dim, 2 * dim)

    def params(self, cond):
        return self.generator(cond).chunk(2, dim=-1)

    def forward(self, x, cond):
        gamma, beta = self.params(cond)
        return (1.0 + gamma) * x + beta


class Identity(nn.Module):

    def forward(self, x, cond):
        return x
