import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3)
        self.conv2 = torch.nn.Conv2d(64, 128, 3)
        self.fc1 = torch.nn.Linear(128 * 6 * 6, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 128 * 6 * 6)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x