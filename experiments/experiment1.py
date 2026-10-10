"""Experiment 1: existing trained MNIST models; no training or project edits."""

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

    output = PROJECT / "results" / "experiment1"
    if output.exists() and any(output.iterdir()):
        output = output / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=True)
    print(f"Results: {output}", flush=True)

    rows = []
    examples = []
    names = []
    for sampler in ["ddpm", "ddim"]:
        for steps in [20, 50, 100]:
            # Warm up at the measured batch size before each configuration.
            generate(model, training_scheduler, noise[:BATCH_SIZE],
                     device_labels[:BATCH_SIZE], sampler, 2, 3)
            images, seconds, timesteps = generate(model, training_scheduler,
                                                  noise, device_labels, sampler, steps, 3)
            predicted, confidence = classifier.predict(images, image_range="zero_one")
            correct = int((predicted == labels).sum())
            row = dict(sampler=sampler, steps=steps, cfg=3, images=100,
                       correct=correct, agreement_percent=correct, seconds=seconds,
                       images_per_second=100 / seconds, seed=SEED,
                       batch_size=BATCH_SIZE, hardware=hardware, precision="float32")
            rows.append(row)
            torch.save(dict(images=images, labels=labels, sampler=sampler,
                            steps=steps, cfg=3, seed=SEED, image_range="zero_one",
                            initial_noise=initial_noise, sampling_seed=SEED + 1,
                            predictions=predicted, confidence=confidence,
                            timesteps=timesteps, checkpoint=str(CHECKPOINT),
                            classifier_checkpoint=str(CLASSIFIER_CHECKPOINT),
                            seconds=seconds, batch_size=BATCH_SIZE, hardware=hardware),
                       output / f"{sampler}_{steps}.pt")
            examples.append(images[:10])  # Fixed first example per digit; no cherry-picking.
            names.append(f"{sampler.upper()} / {steps}")
            print(f"{sampler.upper()} {steps}: {seconds:.3f} s, agreement {correct}/100", flush=True)

    with (output / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(6, 10, figsize=(13, 8))
    for row in range(6):
        for digit in range(10):
            axes[row, digit].imshow(examples[row][digit, 0], cmap="gray", vmin=0, vmax=1)
            axes[row, digit].set_xticks([])
            axes[row, digit].set_yticks([])
            if row == 0:
                axes[row, digit].set_title(str(digit))
            if digit == 0:
                axes[row, digit].set_ylabel(names[row])
    fig.suptitle("DDPM vs DDIM | requested digits 0-9 | identical starting noise | CFG 3")
    fig.tight_layout()
    fig.savefig(output / "ddpm_vs_ddim.png", dpi=160)
    plt.close(fig)

    for metric, ylabel, filename in [
        ("seconds", "Seconds per 100 images", "sampling_time.png"),
        ("agreement_percent", "Requested-digit classifier agreement (%)", "accuracy_vs_steps.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for sampler in ["ddpm", "ddim"]:
            selected = [row for row in rows if row["sampler"] == sampler]
            ax.plot([row["steps"] for row in selected],
                    [row[metric] for row in selected], "o-", label=sampler.upper())
        ax.set(xlabel="Sampling steps", ylabel=ylabel, xticks=[20, 50, 100])
        if metric == "agreement_percent":
            ax.set_ylim(0, 105)
        ax.set_title("One timed run per setting; batch size 20; CFG 3")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)
    print("Experiment 1 complete: 600 evaluation images.", flush=True)


if __name__ == "__main__":
    main()
