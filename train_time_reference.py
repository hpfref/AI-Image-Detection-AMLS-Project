"""
Local CPU train-time reference (PDF Appendix C).

Run once on your development machine to establish a baseline. Per PDF §1.2,
your full local training run must stay within 5x this elapsed time. The grader
evaluates inside a Docker container with --cpus 8.

This script does NOT touch the dataset; it only times forward+backward steps
on synthetic in-memory tensors.
"""

# --- BEGIN: copied from PDF Appendix C ---
import os
import time
import torch
import torch.nn as nn

torch.manual_seed(0)
torch.set_num_threads(min(8, os.cpu_count() or 1))  # max 8 threads
torch.set_num_interop_threads(1)
k = 32
model = nn.Sequential(
    nn.Conv2d(3, k, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(k, k, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(k, 2 * k, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(2 * k, 2 * k, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(2 * k, 4 * k, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(4 * k, 4 * k, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),

    nn.Flatten(),
    nn.Linear(4 * k, 2),
)

x = torch.randn(128, 3, 224, 224)
y = torch.randint(0, 2, (128,))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


def train_steps(steps):
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()


train_steps(10)  # warmup
start = time.perf_counter()
train_steps(35)
elapsed = time.perf_counter() - start
print(f"elapsed_seconds={elapsed:.3f}")
# --- END: copied from PDF Appendix C ---
