from torch import nn


class MLP(nn.Module):
    def __init__(self, layers=[]):
        super().__init__()
        _layers = []
        for i in range(len(layers) - 2):
            _layers.append(nn.Linear(layers[i], layers[i + 1]))
            _layers.append(nn.ReLU())
        _layers.append(nn.Linear(layers[-2], layers[-1]))
        self.layers = nn.Sequential(*_layers)

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    print(MLP()([3]))
