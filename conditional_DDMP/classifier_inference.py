from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


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


CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "mnist_cnn.pth"


class MNISTClassifier:
    def __init__(self, checkpoint=CHECKPOINT):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = Net().to(self.device)
        self.model.load_state_dict(
            torch.load(checkpoint, map_location=self.device, weights_only=True)
        )
        self.model.eval()

    @torch.inference_mode()
    def predict(self, images, image_range="minus_one_one"):
        images = torch.as_tensor(images, dtype=torch.float32)

        if images.ndim == 2:
            images = images.unsqueeze(0).unsqueeze(0)
        elif images.ndim == 3:
            images = images.unsqueeze(1)

        if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
            raise ValueError("Expected images with shape [N, 1, 28, 28]")

        images = images.to(self.device)

        if image_range == "minus_one_one":
            images = (images + 1) / 2
        elif image_range != "zero_one":
            raise ValueError("image_range must be 'minus_one_one' or 'zero_one'")

        images = (images - 0.1307) / 0.3081

        logits = self.model(images)
        probabilities = F.softmax(logits, dim=1)

        confidence, predictions = probabilities.max(dim=1)

        return predictions.cpu(), confidence.cpu()

    def accuracy(self, predictions, labels):
        predictions = torch.as_tensor(predictions)
        labels = torch.as_tensor(labels)

        return (predictions == labels).float().mean().item() * 100

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate saved experiment images with the trained MNIST classifier.")
    parser.add_argument("--images", type=Path, required=True, help="Experiment .pt file containing images, labels and image_range")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()

    saved = torch.load(args.images, map_location="cpu", weights_only=True)
    images = saved["images"]
    labels = saved["labels"]
    if labels.ndim != 1 or len(images) != len(labels) or len(labels) == 0:
        raise ValueError("Expected one requested digit label per image")
    classifier = MNISTClassifier(args.checkpoint)
    predictions, confidence = classifier.predict(images, image_range=saved["image_range"])
    correct = int((predictions == labels).sum())
    print(f"Images: {args.images}")
    print(f"Requested-digit agreement: {correct}/{len(labels)} ({classifier.accuracy(predictions, labels):.2f}%)")
    print(f"Mean classifier confidence: {confidence.mean().item():.4f}")
    if saved.get("cfg") == 0:
        print("CFG 0 ignores requested labels; agreement is not a conditional-generation score.")
    print("Classifier agreement measures label matching, not realism or diversity.")
