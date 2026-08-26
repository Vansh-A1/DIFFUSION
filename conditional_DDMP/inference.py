import os
import argparse
import torch
import matplotlib.pyplot as plt
from diffusers import UNet2DModel, DDPMScheduler

# =========================================================
# Configuration & Setup
# =========================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(script_dir, "checkpoints", "best.pt")
output_dir = os.path.join(script_dir, "samples")
os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# Model & Scheduler Loader
# =========================================================
def load_trained_model(path, device):
    if not os.path.exists(path):
        # Fall back to last.pt if best.pt is missing
        path = os.path.join(script_dir, "checkpoints", "last.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}' or 'last.pt'. Please run train.py first.")

    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config", {})
    num_class_embeds = config.get("num_class_embeds", 11)
    
    # Initialize UNet model matching train.py configuration
    model = UNet2DModel(
        sample_size=28,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        num_class_embeds=num_class_embeds,
        class_embed_type="timestep",
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint.get("epoch", "unknown")
    loss = checkpoint.get("loss", "unknown")
    loss_str = f"{loss:.4f}" if isinstance(loss, float) else str(loss)
    print(f"✓ Loaded model checkpoint from '{os.path.basename(path)}' (Epoch: {epoch}, Loss: {loss_str})")

    # DDPM Noise Scheduler matching train.py setup
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=0.02,
        prediction_type="epsilon"
    )
    
    return model, scheduler

# =========================================================
# Reverse Diffusion Sampling (with Classifier-Free Guidance)
# =========================================================
@torch.no_grad()
def generate_digit(model, scheduler, digit, num_samples=4, cfg_scale=3.0, device="cpu"):
    """
    Generates images of the specified digit (0-9) using reverse diffusion sampling.
    Utilizes Classifier-Free Guidance (CFG) for sharp and accurate digit generation.
    """
    print(f"\n[+] Generating {num_samples} image(s) for Digit '{digit}' (CFG scale: {cfg_scale})...")

    # Target class labels & null token labels (for CFG)
    cond_labels = torch.full((num_samples,), digit, device=device, dtype=torch.long)
    uncond_labels = torch.full((num_samples,), 10, device=device, dtype=torch.long)  # 10 is null token

    # Start from pure standard Gaussian noise
    x = torch.randn(num_samples, 1, 28, 28, device=device)

    # Reverse diffusion sampling loop (1000 steps down to 0)
    for t in scheduler.timesteps:
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)

        if cfg_scale > 1.0:
            # Predict noise for conditional and unconditional inputs
            cond_pred = model(x, t_batch, class_labels=cond_labels).sample
            uncond_pred = model(x, t_batch, class_labels=uncond_labels).sample
            
            # Combine via Classifier-Free Guidance formula
            model_output = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
        else:
            model_output = model(x, t_batch, class_labels=cond_labels).sample

        # Step noise scheduler
        x = scheduler.step(model_output, t, x).prev_sample

    # Rescale from [-1, 1] to [0, 1] range for visualization
    x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0).cpu()
    return x

# =========================================================
# Main Interactive Execution
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Generate specific MNIST digit using trained Conditional Diffusion model.")
    parser.add_argument("--digit", type=int, choices=range(10), help="Digit to generate (0-9)")
    parser.add_argument("--samples", type=int, default=4, help="Number of samples to generate (default: 4)")
    parser.add_argument("--cfg", type=float, default=3.0, help="Classifier-Free Guidance scale (default: 3.0)")
    args = parser.parse_args()

    # If --digit wasn't passed via command line, interactively ask the user
    target_digit = args.digit
    if target_digit is None:
        print("=" * 60)
        print("    MNIST CONDITIONAL DIFFUSION MODEL INFERENCE")
        print("=" * 60)
        while True:
            try:
                user_input = input("Enter the digit you want to produce (0-9): ").strip()
                target_digit = int(user_input)
                if 0 <= target_digit <= 9:
                    break
                else:
                    print("⚠️ Invalid input! Please enter an integer between 0 and 9.")
            except ValueError:
                print("⚠️ Invalid input! Please enter a valid number (0-9).")

    # Load trained model and scheduler
    model, scheduler = load_trained_model(checkpoint_path, device)

    # Generate the requested digit images
    generated_images = generate_digit(
        model=model,
        scheduler=scheduler,
        digit=target_digit,
        num_samples=args.samples,
        cfg_scale=args.cfg,
        device=device
    )

    # Plot & display results
    num_samples = args.samples
    cols = min(num_samples, 4)
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i in range(num_samples):
        axes[i].imshow(generated_images[i].squeeze(), cmap="gray")
        axes[i].set_title(f"Generated Digit: {target_digit}", fontsize=11, fontweight="bold")
        axes[i].axis("off")

    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Conditional Diffusion Model Output | Digit '{target_digit}'", fontsize=13)
    plt.tight_layout()

    save_filepath = os.path.join(output_dir, f"generated_digit_{target_digit}.png")
    plt.savefig(save_filepath, dpi=150)
    print(f"✓ Saved generated image to: {save_filepath}")
    
    #
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()


