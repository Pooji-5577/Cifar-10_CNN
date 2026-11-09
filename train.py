# ---------- Part 1: imports ----------
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import CNN


# ---------- Part 2: settings ----------
BATCH_SIZE = 256
EPOCHS     = 20
LR         = 0.001
DEVICE     = "mps"                              # M4 Mac GPU
DATA_DIR   = "cifar10/cifar-10-batches-py"
CKPT_PATH  = "checkpoints/best.pt"


# ---------- Part 3: data ----------
def load_file(name):
    with open(DATA_DIR + "/" + name, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    images = d[b"data"].reshape(-1, 3, 32, 32)   # (10000, 3072) -> (10000, 3, 32, 32)
    labels = np.array(d[b"labels"])
    return images, labels


def get_loaders():
    # join the 5 files -> 50,000 images
    xs, ys = [], []
    for i in range(1, 6):
        x, y = load_file("data_batch_" + str(i))
        xs.append(x)
        ys.append(y)
    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)

    # shuffle once, then cut 40,000 train / 10,000 validation
    order = np.random.RandomState(42).permutation(50000)
    x_all, y_all = x_all[order], y_all[order]
    x_train, y_train = x_all[:40000], y_all[:40000]
    x_val,   y_val   = x_all[40000:], y_all[40000:]

    # separate test file -> 10,000 images
    x_test, y_test = load_file("test_batch")

    def to_dataset(x, y):
        x = torch.tensor(x, dtype=torch.float32) / 255.0   # pixels 0-255 -> 0-1
        y = torch.tensor(y, dtype=torch.long)
        return TensorDataset(x, y)

    train_loader = DataLoader(to_dataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(to_dataset(x_val,   y_val),   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(to_dataset(x_test,  y_test),  batch_size=BATCH_SIZE)
    return train_loader, val_loader, test_loader


# ---------- Part 4: one pass over a loader (trains if given an optimizer) ----------
def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()      # dropout on for training, off otherwise

    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:                # one batch at a time
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)                  # model guesses the batch
            loss = loss_fn(outputs, labels)          # how wrong the guesses are

            if training:
                optimizer.zero_grad()                # clear old corrections
                loss.backward()                      # work out new corrections
                optimizer.step()                     # fix the weights

            total_loss += loss.item() * labels.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            count      += labels.size(0)

    return total_loss / count, correct / count       # average loss, accuracy


if __name__ == "__main__":
    torch.manual_seed(42)
    train_loader, val_loader, test_loader = get_loaders()

    model     = CNN().to(DEVICE)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, alpha=0.9, eps=1e-7)

    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Part 4: train on the 40,000
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer)

        # Part 5: validate on the 10,000 (no learning)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn)

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f}")

        # Part 6: save the model if it is the best so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CKPT_PATH)
            print("   saved new best model")

    # Part 7: test once, using the best saved model
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    test_loss, test_acc = run_epoch(model, test_loader, loss_fn)
    print(f"Test loss {test_loss:.4f} | Test accuracy {test_acc:.4f}")