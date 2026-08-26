import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

# =========================================================
# Configuration
# =========================================================
class Config:
    epochs = 30
    batch_size = 128
    learning_rate = 2e-4
    num_timesteps = 1000
    beta_schedule = "linear"
    
    # Model architecture params
    image_size = 28
    in_channels = 1
    out_channels = 1
    num_classes = 10         # MNIST digits 0-9
    num_class_embeds = 11    # 0-9 digits + 1 null token for Classifier-Free Guidance (CFG)
    cfg_drop_prob = 0.1      # 10% probability to drop condition during training
    
    # Output directories
    output_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    samples_dir = os.path.join(output_dir, "samples")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# Setup Directories
# =========================================================
os.makedirs(Config.checkpoint_dir, exist_ok=True)
os.makedirs(Config.samples_dir, exist_ok=True)


# =========================================================
# Data Loading (MNIST)
# =========================================================
def get_dataloader(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0)  # Scale images to [-1, 1]
    ])
    
    train_dataset = datasets.MNIST(
        root=os.path.join(Config.output_dir, "data"),
        train=True,
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    return dataloader


# =========================================================
# Model & Scheduler Initialization using HuggingFace `diffusers`
# =========================================================
def build_model_and_scheduler():
    # Using `UNet2DModel` from diffusers with class embedding support
    model = UNet2DModel(
        sample_size=Config.image_size,
        in_channels=Config.in_channels,
        out_channels=Config.out_channels,
        layers_per_block=2,
        block_out_channels=(64, 128, 256),
        down_block_types=(
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
        num_class_embeds=Config.num_class_embeds,
        class_embed_type="timestep",
    ).to(Config.device)

    # Standard DDPM noise scheduler from diffusers
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=Config.num_timesteps,
        beta_schedule=Config.beta_schedule,
        beta_start=1e-4,
        beta_end=0.02,
        prediction_type="epsilon"
    )
    
    return model, noise_scheduler


# =========================================================
# Validation Sampling (Visualizing Class-Conditioned Generation)
# =========================================================
@torch.no_grad()
def save_sample_grid(model, noise_scheduler, epoch):
    model.eval()
    
    # Generate 1 sample for each digit 0-9 (repeated twice for 2 rows of 0-9 = 20 images)
    num_per_digit = 2
    labels = torch.tensor([digit for digit in range(10) for _ in range(num_per_digit)], device=Config.device)
    num_samples = labels.shape[0]
    
    # Pure Gaussian noise start
    x = torch.randn(num_samples, Config.in_channels, Config.image_size, Config.image_size, device=Config.device)
    
    # Reverse diffusion sampling
    for t in noise_scheduler.timesteps:
        t_batch = torch.full((num_samples,), t, device=Config.device, dtype=torch.long)
        
        # Model noise prediction conditioned on class labels
        model_output = model(x, t_batch, class_labels=labels).sample
        
        # Step reverse scheduler
        x = noise_scheduler.step(model_output, t, x).prev_sample

    # Rescale from [-1, 1] to [0, 1]
    x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
    
    # Save visual grid
    fig, axes = plt.subplots(2, 10, figsize=(15, 3.5))
    for idx in range(num_samples):
        r = idx // 10
        c = idx % 10
        axes[r, c].imshow(x[idx].cpu().squeeze(), cmap="gray")
        if r == 0:
            axes[r, c].set_title(f"Digit {labels[idx].item()}")
        axes[r, c].axis("off")
        
    plt.suptitle(f"Conditional DDPM Samples | Epoch {epoch}", fontsize=14)
    plt.tight_layout()
    sample_path = os.path.join(Config.samples_dir, f"epoch_{epoch:03d}.png")
    plt.savefig(sample_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved validation sample grid: {sample_path}")
    
    model.train()


# =========================================================
# Main Training Function
# =========================================================
def train():
    print("=" * 65)
    print("Starting Conditional Diffusion Training (using HF diffusers)")
    print(f"Device: {Config.device} | Epochs: {Config.epochs} | Batch Size: {Config.batch_size}")
    print("=" * 65)
    
    dataloader = get_dataloader(Config.batch_size)
    model, noise_scheduler = build_model_and_scheduler()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=1e-4)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * Config.epochs
    )
    loss_fn = nn.MSELoss()
    
    best_loss = float("inf")

    for epoch in range(1, Config.epochs + 1):
        model.train()
        running_loss = 0.0
        
        for step, (images, labels) in enumerate(dataloader):
            images = images.to(Config.device)
            labels = labels.to(Config.device)
            
            # Classifier-Free Guidance (CFG): randomly drop class label with probability cfg_drop_prob
            if Config.cfg_drop_prob > 0.0:
                drop_mask = torch.rand(labels.shape[0], device=Config.device) < Config.cfg_drop_prob
                # Replace dropped labels with null label index (10)
                labels = torch.where(drop_mask, torch.tensor(10, device=Config.device), labels)
            
            # Sample random noise & timesteps for batch
            noise = torch.randn_like(images)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (images.shape[0],),
                device=Config.device
            ).long()
            
            # Add noise to clean images according to schedule
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            
            # Predict noise conditioned on timestep AND digit class label
            model_pred = model(noisy_images, timesteps, class_labels=labels).sample
            
            # Compute MSE loss
            loss = loss_fn(model_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch:02d}/{Config.epochs:02d}] - Loss: {avg_loss:.5f} | LR: {current_lr:.2e}")
        
        # Save checkpoints
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "config": {
                "num_classes": Config.num_classes,
                "num_class_embeds": Config.num_class_embeds,
                "num_timesteps": Config.num_timesteps,
                "image_size": Config.image_size
            }
        }
        
        # Save last checkpoint
        torch.save(checkpoint_data, os.path.join(Config.checkpoint_dir, "last.pt"))
        
        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint_data, os.path.join(Config.checkpoint_dir, "best.pt"))
            print(f"  ✓ Saved new best checkpoint (Loss: {best_loss:.5f})")

        # Save validation sample grid every 5 epochs
        if epoch % 5 == 0 or epoch == Config.epochs:
            save_sample_grid(model, noise_scheduler, epoch)

    print("\nTraining completed successfully!")
    print(f"Checkpoints saved in: {Config.checkpoint_dir}")


if __name__ == "__main__":
    train()

