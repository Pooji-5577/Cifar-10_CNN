# ---------- Part 1: imports ----------
import torch
import torch.nn as nn


# ---------- Part 2: the layers ----------
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Block 1: 3x32x32 -> 32x32x32 -> 32x30x30 -> 32x15x15
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),   # conv2d_13: 896 params
            nn.Conv2d(32, 32, kernel_size=3, padding=0), nn.ReLU(),  # conv2d_14: 9,248
            nn.MaxPool2d(2),                                         # 30 -> 15
            nn.Dropout(0.25),
        )

        # Block 2: 32x15x15 -> 64x15x15 -> 64x13x13 -> 64x6x6
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),  # conv2d_15: 18,496
            nn.Conv2d(64, 64, kernel_size=3, padding=0), nn.ReLU(),  # conv2d_16: 36,928
            nn.MaxPool2d(2),                                         # 13 -> 6
            nn.Dropout(0.25),
        )

        # Block 3: 64x6x6 -> 64x6x6 -> 64x4x4 -> 64x2x2
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),  # conv2d_17: 36,928
            nn.Conv2d(64, 64, kernel_size=3, padding=0), nn.ReLU(),  # conv2d_18: 36,928
            nn.MaxPool2d(2),                                         # 4 -> 2
            nn.Dropout(0.25),
        )

        # Head: 64x2x2 = 256 -> 512 -> 10
        self.head = nn.Sequential(
            nn.Flatten(),                    # flatten_3: 256 numbers
            nn.Linear(256, 512), nn.ReLU(),  # dense_4: 131,584
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),     # dense_5: 5,130
        )

    # ---------- Part 3: the order data flows ----------
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)


# ---------- Part 4: self-check (run: python model.py) ----------
if __name__ == "__main__":
    model = CNN()
    out = model(torch.randn(1, 3, 32, 32))
    total = sum(p.numel() for p in model.parameters())
    print("output shape:", tuple(out.shape))   # expect (1, 10)
    print("total params:", f"{total:,}")       # expect 276,138