import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from diffusion_scratch import UNet

# =========================================================
# Configuration
# =========================================================
num_timesteps = 1000  # Must match training schedule
beta_start = 1e-4
beta_end = 0.02
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "checkpoints/best.pt"

os.makedirs("samples", exist_ok=True)

# =========================================================
# Precalculate DDPM reverse diffusion parameters
# =========================================================
betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
alphas = 1.0 - betas
alphas_bars = torch.cumprod(alphas, dim=0).to(device)
alphas_bars_prev = torch.cat([torch.tensor([1.0], device=device), alphas_bars[:-1]])

# Posterior variance: \tilde{\beta}_t = \beta_t * (1 - \bar{\alpha}_{t-1}) / (1 - \bar{\alpha}_t)
posterior_variance = betas * (1.0 - alphas_bars_prev) / (1.0 - alphas_bars)

# =========================================================
# Load Model
# =========================================================
def load_trained_model(path, device):
    model = UNet(in_channels=1, out_channels=1, base_channels=64, time_emb_dim=128).to(device)
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "unknown")
        loss = checkpoint.get("loss", "unknown")
        loss_str = f"{loss:.4f}" if isinstance(loss, float) else str(loss)
        print(f"✓ Loaded checkpoint '{path}' (Epoch {epoch}, Loss: {loss_str})")
    else:
        raise FileNotFoundError(f"Checkpoint not found at '{path}'")
    model.eval()
    return model

# =========================================================
# Sampling function (Reverse Diffusion)
# =========================================================
@torch.no_grad()
def sample_ddpm(model, num_samples=16, save_trajectory_steps=None):
    """
    Generate samples starting from pure Gaussian noise x_T ~ N(0, I)
    using DDPM reverse sampling.
    """
    # 1. Start from pure random noise
    x = torch.randn(num_samples, 1, 28, 28, device=device)
    
    trajectory = {}
    
    # 2. Iterate backwards from t = T-1 down to 0
    for t in reversed(range(num_timesteps)):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        
        # Predict noise added at timestep t
        predicted_noise = model(x, t_batch)
        
        alpha_t = alphas[t]
        alpha_bar_t = alphas_bars[t]
        beta_t = betas[t]
        
        # Mean formula: \mu_t = (1 / \sqrt{\alpha_t}) * (x_t - (\beta_t / \sqrt{1 - \bar{\alpha}_t}) * \hat{\epsilon})
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise
        )
        
        if t > 0:
            noise = torch.randn_like(x)
            # Use \sqrt{\beta_t} or \sqrt{\tilde{\beta}_t} for variance
            sigma_t = torch.sqrt(beta_t)
            x = mean + sigma_t * noise
        else:
            x = mean
            
        if save_trajectory_steps and t in save_trajectory_steps:
            trajectory[t] = x.detach().cpu()
            
    # Rescale from [-1, 1] to [0, 1] for visualization
    x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
    return x.cpu(), trajectory

# =========================================================
# Main Inference Execution
# =========================================================
if __name__ == "__main__":
    print(f"Running DDPM Sampling on device: {device}")
    model = load_trained_model(checkpoint_path, device)
    
    num_samples = 16
    timesteps_to_save = [999, 750, 500, 250, 100, 0]
    
    print(f"Generating {num_samples} samples using reverse diffusion ({num_timesteps} steps)...")
    samples, trajectory = sample_ddpm(model, num_samples=num_samples, save_trajectory_steps=timesteps_to_save)
    
    # ---------------------------------------------------------
    # Save Generated Digits Grid (4x4)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(16):
        ax = axes[i // 4, i % 4]
        ax.imshow(samples[i].squeeze(), cmap="gray")
        ax.axis("off")
    plt.suptitle("DDPM Generated Handwritten Digits (MNIST)", fontsize=14)
    plt.tight_layout()
    grid_path = "samples/generated_digits_grid.png"
    plt.savefig(grid_path, dpi=150)
    plt.close()
    print(f"✓ Saved generated digits grid to: {grid_path}")
    
    # ---------------------------------------------------------
    # Save Reverse Sampling Progress Breakdown
    # ---------------------------------------------------------
    fig, axes = plt.subplots(4, len(timesteps_to_save), figsize=(14, 8))
    for row in range(4):
        for col, t_step in enumerate(timesteps_to_save):
            img = trajectory[t_step][row].squeeze()
            img = torch.clamp((img + 1.0) / 2.0, 0.0, 1.0)
            ax = axes[row, col]
            ax.imshow(img, cmap="gray")
            if row == 0:
                ax.set_title(f"t = {t_step}")
            ax.axis("off")
    plt.suptitle("DDPM Reverse Sampling Process (Noise -> Digit)", fontsize=14)
    plt.tight_layout()
    traj_path = "samples/sampling_progress.png"
    plt.savefig(traj_path, dpi=150)
    plt.close()
    print(f"✓ Saved sampling process breakdown to: {traj_path}")

