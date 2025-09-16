from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.1,
        batchnorm: bool = True,
        momentum: float = 0.01,
        eps=1e-3,
    ):
        super().__init__()
        if len(hidden_dims) != 0:
            model = [
                nn.Linear(in_dim, hidden_dims[0]),
                nn.BatchNorm1d(
                    hidden_dims[0],
                    momentum=momentum,
                    eps=eps,
                )
                if batchnorm
                else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]

            for ndim in range(len(hidden_dims) - 1):
                model += [
                    nn.Linear(hidden_dims[ndim], hidden_dims[ndim + 1]),
                    nn.BatchNorm1d(
                        hidden_dims[ndim + 1],
                        momentum=momentum,
                        eps=eps,
                    )
                    if batchnorm
                    else nn.Identity(),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]

            model += [
                nn.Linear(hidden_dims[-1], out_dim),
                nn.BatchNorm1d(
                    out_dim,
                    momentum=momentum,
                    eps=eps,
                )
                if batchnorm
                else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        else:
            model = [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(
                    out_dim,
                    momentum=momentum,
                    eps=eps,
                )
                if batchnorm
                else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
