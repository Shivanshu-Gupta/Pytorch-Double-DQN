import torch.nn as nn
from IPython.core.debugger import Tracer; debug_here = Tracer()


class DQNMLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQNMLPModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.mlp = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU())
        self.fc_out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x = x.view(x.size(0), self.input_size)
        h = self.mlp(x)
        return self.fc_out(h)


class DQNCNNModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNCNNModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.cnn = nn.Sequential(
            nn.Conv2d(self.input_size, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_size))

    def forward(self, x):
        # assert(list(x.size()[-3]) == self.input_size)
        # debug_here()
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
