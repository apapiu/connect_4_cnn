import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA GPU (CUDA)")
else:
    device = torch.device("cpu")
    print("Using CPU")

print(f"Training on: {device}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ResBlock(nn.Module):
    def __init__(self, c_in, c_mid, c_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_mid, 3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(c_mid),
            nn.Conv2d(c_mid, c_out, 3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(c_out),
        )

        self.resid = (
            nn.Conv2d(c_in, c_out, 1, padding="same")
            if c_in != c_out
            else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x) + self.resid(x)


class ResTower(nn.Module):
    def __init__(self, c_in, img_dim):
        super().__init__()
        self.tower = nn.Sequential(
            ResBlock(3, 1 * c_in, 1 * c_in),
            ResBlock(1 * c_in, 1 * c_in, 1 * c_in),
            nn.MaxPool2d(2),
            ResBlock(1 * c_in, 2 * c_in, 2 * c_in),
            ResBlock(2 * c_in, 2 * c_in, 2 * c_in),
            nn.MaxPool2d(2),
            ResBlock(2 * c_in, 4 * c_in, 4 * c_in),
            ResBlock(4 * c_in, 4 * c_in, 4 * c_in),
            nn.AvgPool2d(img_dim // 4),  # 256 dim for 32 input
            nn.Flatten(),
            nn.Linear(4 * c_in, 4 * c_in),
            nn.ReLU(),
            nn.Linear(4 * c_in, 10),
        )

    def forward(self, x):
        return self.tower(x)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.global_step = 0

        self.conv = nn.Sequential(
            nn.Conv2d(1, 128, 4, padding="same"),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding="same"),
            nn.ReLU(),
            # nn.Conv2d(256, 256, 3, padding='same'),
            # nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0003)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10000, gamma=0.1
        )

        # Use mixed precision only for CUDA (not supported on MPS yet)
        self.use_amp = device.type == "cuda"
        if self.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None

    def forward(self, x):
        x = self.conv(x)
        return x

    def weighted_mse_loss(self, input, target, weight):
        return (weight * (input - target) ** 2).mean()

    def train_step(self, x, y, weight):
        if self.use_amp:
            # Use mixed precision for CUDA
            with autocast():
                preds = self(x)
                y = y.view(-1, 1)
                weight = weight.view(-1, 1)
                loss = self.weighted_mse_loss(preds, y, weight)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard training for MPS/CPU (Apple Silicon optimized)
            preds = self(x)
            y = y.view(-1, 1)
            weight = weight.view(-1, 1)
            loss = self.weighted_mse_loss(preds, y, weight)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()
        self.global_step += 1

        return loss.item()
