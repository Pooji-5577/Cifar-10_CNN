# CIFAR-10 CNN in PyTorch

A small convolutional neural network (CNN) built from scratch in PyTorch to classify CIFAR-10 images (32x32 color images, 10 classes). The architecture follows the CNN on page 147 of *Deep Learning Using Python*.

## Results

| Set | Images | Accuracy |
|---|---|---|
| Train (epoch 20) | 40,000 | 0.7342 |
| Validation (best, epoch 19) | 10,000 | 0.7300 |
| Test (best saved model, scored once) | 10,000 | 0.7260 |

Random guessing on 10 classes scores about 0.10. Accuracy is a fraction from 0 to 1, so 0.7260 means 7,260 of 10,000 test images were right.

Train, validation and test accuracy are close, so the model is not overfitting yet. Validation accuracy was still improving near epoch 20.

## Files

| File | What it does |
|---|---|
| `model.py` | The `CNN` class. Run it alone for a self-check. |
| `train.py` | Settings, data loading, training loop, validation, saving the best model, final test. |
| `checkpoints/best.pt` | Weights of the best epoch, made by `train.py`. |
| `cifar10/cifar-10-batches-py/` | The CIFAR-10 python version (pickle files). |

## Data split

- `data_batch_1` to `data_batch_5`: 50,000 images, shuffled once with seed 42, then cut into 40,000 train and 10,000 validation.
- `test_batch`: 10,000 images, used only once at the end.
- Pixels are scaled from 0-255 to 0-1. Batches hold 256 images.

## Model

3 blocks of (conv, conv, max-pool, dropout 0.25), then flatten, dense 512, dropout 0.5, dense 10.

| Layer | Output shape |
|---|---|
| Input | (3, 32, 32) |
| Block 1 (32 filters) | (32, 15, 15) |
| Block 2 (64 filters) | (64, 6, 6) |
| Block 3 (64 filters) | (64, 2, 2) |
| Flatten | 256 |
| Dense | 512 |
| Output | 10 |

Total parameters: 276,138 (matches the book).

## Training settings

| Setting | Value |
|---|---|
| Batch size | 256 |
| Epochs | 20 |
| Optimizer | RMSprop, learning rate 0.001, alpha 0.9, eps 1e-7 |
| Loss | CrossEntropyLoss |
| Device | `mps` (Mac GPU) |
| Seed | 42 |

## How to run

From the project folder:

```
./venv/bin/python model.py    # self-check: output shape (1, 10), 276,138 params
./venv/bin/python train.py    # trains 20 epochs, saves checkpoints/best.pt, prints test accuracy
```

## Ideas to improve (one change at a time, compare with 0.7260)

1. More epochs (for example 50, as in the book)
2. Data augmentation (random flips and shifts)
3. Batch normalization
4. Learning rate schedule
5. A bigger or deeper model
