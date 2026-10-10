from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "data"
CHECKPOINT_PATH = SCRIPT_DIR / "checkpoints" / "mnist_cnn.pth"
CHECKPOINT_PATH.parent.mkdir(exist_ok=True)

full_train_data = datasets.MNIST(
    root=DATA_PATH,
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root=DATA_PATH,
    train=False,
    download=True,
    transform=transform
)

train_data, val_data = random_split(
    full_train_data,
    [54000, 6000],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_data,
    batch_size=1000
)

test_loader = DataLoader(
    test_data,
    batch_size=1000
)

class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x)), 2)

        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.flatten(1)

        x = self.dropout(F.relu(self.fc1(x)))

        return self.fc2(x)


model = Net().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

criterion = nn.CrossEntropyLoss()
def train():

    model.train()

    total_loss = 0

    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        output = model(x)

        loss = criterion(output, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
       for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            predictions = output.argmax(dim=1)
            correct += (predictions == y).sum().item()
            total += y.size(0)
    return 100 * correct / total

epochs = 20
best_val_accuracy = 0
for epoch in range(1, epochs + 1):
    train_loss = train()
    val_accuracy = evaluate(val_loader)
    print(
        f"Epoch {epoch}/{epochs} | "
        f"Loss: {train_loss:.4f} | "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(
            model.state_dict(),
            CHECKPOINT_PATH
        )
        print("Best model saved!")
print("\nTraining completed.")
model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )
)

test_accuracy = evaluate(test_loader)

print(f"\nBest Validation Accuracy: {best_val_accuracy:.2f}%")
print(f"Final Test Accuracy: {test_accuracy:.2f}%")

print(f"\nModel saved to {CHECKPOINT_PATH}")