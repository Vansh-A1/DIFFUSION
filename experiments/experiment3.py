"""Experiment 3: existing trained MNIST models; no training or project edits."""

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

# Import the existing model and classifier from conditional_DDMP.
sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "conditional_DDMP"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from diffusers import DDPMScheduler, DDIMScheduler
from inference import load_trained_model
from classifier_inference import MNISTClassifier

SEED = 42
BATCH_SIZE = 20
CHECKPOINT = PROJECT / "conditional_DDMP/checkpoints/best.pt"
CLASSIFIER_CHECKPOINT = PROJECT / "conditional_DDMP/checkpoints/mnist_cnn.pth"


@torch.inference_mode()
def generate(model, training_scheduler, noise, labels, sampler, steps, cfg):
    """Use your sampling updates, with supplied noise for a fair comparison."""
    if sampler == "ddpm":
        scheduler = DDPMScheduler.from_config(training_scheduler.config)
    else:
        assert sampler == "ddim"
        scheduler = DDIMScheduler.from_config(training_scheduler.config)
    scheduler.set_timesteps(steps, device=noise.device)
    assert len(scheduler.timesteps) == steps
    assert scheduler.config.prediction_type == "epsilon"
    # A separate generator controls DDPM's additional reverse-process noise.
    generator = torch.Generator(device=noise.device).manual_seed(SEED + 1)
    batches = []

    if noise.is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for offset in range(0, len(labels), BATCH_SIZE):
        x = noise[offset:offset + BATCH_SIZE].clone()
        requested = labels[offset:offset + BATCH_SIZE]
        null_labels = torch.full_like(requested, 10)
        for t in scheduler.timesteps:
            t_batch = t.expand(len(x))
            # CFG 0 is unconditional; CFG 1 uses the requested digit directly.
            if cfg == 0:
                prediction = model(x, t_batch, class_labels=null_labels).sample
            elif cfg == 1:
                prediction = model(x, t_batch, class_labels=requested).sample
            else:
                conditional = model(x, t_batch, class_labels=requested).sample
                unconditional = model(x, t_batch, class_labels=null_labels).sample
                prediction = unconditional + cfg * (conditional - unconditional)
            if sampler == "ddpm":
                x = scheduler.step(prediction, t, x, generator=generator).prev_sample
            else:
                # Preserve your DDIM settings, including the clipping correction.
                x = scheduler.step(prediction, t, x, eta=0.0,
                                   use_clipped_model_output=True,
                                   generator=generator).prev_sample
        batches.append(x)
    if noise.is_cuda:
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start

    # Conversion, classifier evaluation, plots and saving are OUTSIDE timing.
    images = ((torch.cat(batches) + 1.0) / 2.0).clamp(0, 1).cpu()
    assert images.shape == (len(labels), 1, 28, 28)
    assert torch.isfinite(images).all() and 0 <= images.min() <= images.max() <= 1
    return images, seconds, scheduler.timesteps.cpu()


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hardware = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"Hardware: {hardware}; batch size: {BATCH_SIZE}; FP32; seed: {SEED}", flush=True)
    assert CHECKPOINT.is_file() and CLASSIFIER_CHECKPOINT.is_file()
    model, training_scheduler = load_trained_model(str(CHECKPOINT), device)
    classifier = MNISTClassifier(str(CLASSIFIER_CHECKPOINT))
    model.eval()
    classifier.model.eval()

    # Ten rows of digits 0-9: exactly ten requested samples per digit.
    labels = torch.arange(10).repeat(10)
    assert torch.equal(torch.bincount(labels), torch.full((10,), 10))
    initial_noise = torch.randn(100, 1, 28, 28,
                               generator=torch.Generator().manual_seed(SEED))
    noise = initial_noise.to(device)
    device_labels = labels.to(device)

    # Tiny preflight: both schedulers, range/shape, normalization and CFG 0.
    for sampler in ["ddpm", "ddim"]:
        tiny, _, _ = generate(model, training_scheduler, noise[:2],
                              device_labels[:2], sampler, 2, 3)
        predicted, _ = classifier.predict(tiny, image_range="zero_one")
        normalized = (tiny.to(device) - 0.1307) / 0.3081
        direct = classifier.model(normalized).argmax(1).cpu()
        assert torch.equal(predicted, direct)
    uncond, _, _ = generate(model, training_scheduler, noise[:2],
                            device_labels[:2], "ddim", 2, 0)
    changed, _, _ = generate(model, training_scheduler, noise[:2],
                             (device_labels[:2] + 5) % 10, "ddim", 2, 0)
    assert torch.equal(uncond, changed), "CFG 0 must ignore requested digits"
    print("Preflight passed: checkpoints, schedulers, image range, classifier normalization, CFG 0.", flush=True)

    output = PROJECT / "results" / "experiment3"
    if output.exists() and any(output.iterdir()):
        output = output / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=True)
    print(f"Results: {output}", flush=True)

    from torchvision.datasets import MNIST
    dataset = MNIST(root=str(PROJECT / "conditional_DDMP/data"),
                    train=False, download=False)
    # Randomly select ten test images for each digit, using a fixed seed.
    real_indices = []
    selection_rng = torch.Generator().manual_seed(SEED)
    for digit in range(10):
        available = torch.where(dataset.targets == digit)[0]
        selected = available[torch.randperm(len(available), generator=selection_rng)[:10]]
        real_indices.append(selected)
    # Reorder to the same ten rows of 0-9 as the generated samples.
    real_indices = torch.stack(real_indices, dim=1).flatten()
    real_images = dataset.data[real_indices].unsqueeze(1).float() / 255.0
    real_labels = dataset.targets[real_indices]
    assert torch.equal(real_labels, labels)
    assert real_images.shape == (100, 1, 28, 28)

    generate(model, training_scheduler, noise[:BATCH_SIZE],
             device_labels[:BATCH_SIZE], "ddim", 2, 3)
    images, seconds, timesteps = generate(model, training_scheduler,
                                          noise, device_labels, "ddim", 50, 3)
    real_predicted, _ = classifier.predict(real_images, image_range="zero_one")
    predicted, confidence = classifier.predict(images, image_range="zero_one")
    real_correct = int((real_predicted == real_labels).sum())
    generated_correct = int((predicted == labels).sum())
    rows = [
        dict(group="real_test", images=100, correct=real_correct, accuracy_percent=real_correct,
             metric="accuracy against true test labels", sampler="none", steps=0, cfg="",
             generation_seconds="", seed=SEED, batch_size=BATCH_SIZE, hardware=hardware),
        dict(group="generated", images=100, correct=generated_correct, accuracy_percent=generated_correct,
             metric="agreement with requested labels", sampler="ddim", steps=50, cfg=3,
             generation_seconds=seconds, seed=SEED, batch_size=BATCH_SIZE, hardware=hardware),
    ]
    with (output / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    # One generated configuration file; keep the exact selected test subset here too.
    torch.save(dict(images=images, labels=labels, sampler="ddim", steps=50, cfg=3,
                    seed=SEED, image_range="zero_one", initial_noise=initial_noise,
                    sampling_seed=SEED + 1, predictions=predicted, confidence=confidence,
                    timesteps=timesteps, checkpoint=str(CHECKPOINT),
                    classifier_checkpoint=str(CLASSIFIER_CHECKPOINT), seconds=seconds,
                    batch_size=BATCH_SIZE, hardware=hardware, real_images=real_images,
                    real_labels=real_labels, real_predictions=real_predicted,
                    real_test_indices=real_indices), output / "ddim_50_cfg3.pt")

    # Show the first three examples per digit, alternating real and generated rows.
    fig, axes = plt.subplots(6, 10, figsize=(13, 8))
    for example in range(3):
        for group, source in enumerate([real_images, images]):
            row = 2 * example + group
            for digit in range(10):
                index = 10 * example + digit
                axes[row, digit].imshow(source[index, 0], cmap="gray", vmin=0, vmax=1)
                axes[row, digit].set_xticks([])
                axes[row, digit].set_yticks([])
                if row == 0:
                    axes[row, digit].set_title(str(digit))
                if digit == 0:
                    axes[row, digit].set_ylabel("Real" if group == 0 else "Generated")
    fig.suptitle("Real MNIST test images vs generated digits | DDIM 50 / CFG 3")
    fig.tight_layout()
    fig.savefig(output / "real_vs_generated.png", dpi=160)
    plt.close(fig)
    print(f"Real test subset: {real_correct}/100; generated: {generated_correct}/100", flush=True)
    print("Experiment 3 complete: 100 generated images and 100 real test images.", flush=True)
    print("Classifier agreement does not establish image realism or rule out memorization.", flush=True)


if __name__ == "__main__":
    main()
