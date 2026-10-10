import argparse
import os

import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image
from diffusers import UNet2DModel, DDIMScheduler


parser = argparse.ArgumentParser(description="Generate an MNIST digit using DDIM.")
parser.add_argument("--digit", type=int, choices=range(10), help="Digit to generate (0-9)")
parser.add_argument("--steps", type=int, default=250, help="DDIM denoising steps, 1-1000 (default: 250)")
parser.add_argument("--cfg", type=float, default=3.0, help="Classifier-free guidance scale (default: 3.0)")
parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
parser.add_argument("--no-show", action="store_true", help="Save without opening a plot")
args = parser.parse_args()
if not 1 <= args.steps <= 1000:
    parser.error("--steps must be between 1 and 1000, matching the training schedule.")
if not torch.isfinite(torch.tensor(args.cfg)).item() or args.cfg < 0:
    parser.error("--cfg must be a finite, non-negative number.")

user_input_digit = args.digit
while user_input_digit is None:
    try:
        value = int(input("Enter the digit you want to generate (0-9): ").strip())
        if 0 <= value <= 9:
            user_input_digit = value
        else:
            print("Please enter a digit from 0 to 9.")
    except ValueError:
        print("Please enter an integer digit from 0 to 9.")
    except EOFError:
        parser.error("No interactive input available; pass --digit with a value from 0 to 9.")
# ---------------------------
# Settings
# ---------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(script_dir, "checkpoints", "best.pt")
out_dir = os.path.join(script_dir, "samples")
os.makedirs(out_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
digit = user_input_digit
guidance_scale = args.cfg
num_samples = 1
null_class = 10
num_inference_steps = args.steps
eta = 0.0

# ---------------------------
# Rebuild model architecture
# ---------------------------
model = UNet2DModel(
    sample_size=28,
    in_channels=1,
    out_channels=1,
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
    num_class_embeds=11,
    class_embed_type="timestep",
).to(device)

# ---------------------------
# Load learned weights
# ---------------------------
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Loaded {checkpoint_path} (epoch {checkpoint.get('epoch', 'unknown')})")
del checkpoint

# ---------------------------
# DDIM scheduler
# Same training noise schedule, different sampler
# ---------------------------
scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_schedule="linear",
    beta_start=1e-4,
    beta_end=0.02,
    prediction_type="epsilon",
)
scheduler.set_timesteps(num_inference_steps, device=device)

# ---------------------------
# Starting noise and labels
# ---------------------------
generator = torch.Generator(device=device).manual_seed(args.seed)
x = torch.randn(
    num_samples,
    1,
    28,
    28,
    device=device,
    generator=generator,
)
class_labels = torch.full(
    (num_samples,), digit, device=device, dtype=torch.long,
)
null_labels = torch.full(
    (num_samples,), null_class, device=device, dtype=torch.long,
)

# ---------------------------
# DDIM reverse process
# ---------------------------
print(
    f"Generating digit {digit} on {device} with {num_inference_steps} DDIM steps "
    f"(CFG {guidance_scale}, seed {args.seed})..."
)
with torch.inference_mode():
    for t in scheduler.timesteps:
        t_batch = torch.full(
            (num_samples,),
            t.item(),
            device=device,
            dtype=torch.long,
        )
        epsilon_cond = model(x, t_batch, class_labels=class_labels).sample
        epsilon_uncond = model(x, t_batch, class_labels=null_labels).sample
        epsilon_guided = epsilon_uncond + guidance_scale * (
            epsilon_cond - epsilon_uncond
        )
        x = scheduler.step(
            epsilon_guided,
            t,
            x,
            eta=eta,
            # Keep the noise direction consistent with the clipped clean-image
            # estimate. Without this, guidance can distort the digit even with
            # more denoising steps.
            use_clipped_model_output=True,
            generator=generator,
        ).prev_sample

# ---------------------------
# Save image
# ---------------------------
image = ((x + 1.0) / 2.0).clamp(0.0, 1.0).cpu()
output_path = os.path.join(out_dir, f"ddim_digit_{digit}.png")
save_image(image, output_path)
print(f"Saved DDIM sample to: {output_path}")

if not args.no_show:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image[0, 0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"DDIM generated digit: {digit}")
    ax.axis("off")
    fig.tight_layout()
    plt.show()
    plt.close(fig)
