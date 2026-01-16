import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual # The "Skip Connection"
        return F.relu(out)

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.start = nn.Sequential(
            nn.Conv2d(19, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # 5 Residual Blocks for deep pattern recognition
        self.res_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(5)])
        
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Tanh() # Outputs -1 (Black wins) to 1 (White wins)
        )

    def forward(self, x):
        x = self.start(x)
        x = self.res_blocks(x)
        return self.value_head(x)