# Diffusion Models: From Noise to Images

**Study DDPMs through an explicit PyTorch implementation and a class-conditioned MNIST workflow.**

This repository explores the forward noising process, noise prediction, and reverse sampling. It contains two complementary tracks: an unconditional model with its core components implemented directly, and a conditional model using Hugging Face Diffusers.

## Choose a track

| Track | Code | Focus |
| --- | --- | --- |
| Unconditional DDPM | [`Unconditional_DDPM/`](Unconditional_DDPM/) | Custom timestep embeddings, U-Net components, diffusion schedule, and noise-prediction training |
| Conditional DDPM | [`conditional_DDMP/`](conditional_DDMP/) | MNIST class conditioning and classifier-free guidance using `UNet2DModel` and `DDPMScheduler` |

Both use MNIST digit images. The unconditional track implements the core diffusion machinery directly; the conditional track builds on Diffusers.

## Core idea

```text
Clean image x0 + sampled noise + timestep t
                  ↓
              Noisy image xt
                  ↓
          U-Net predicts the noise
                  ↓
       MSE(predicted noise, sampled noise)
```

The forward process uses cumulative noise-schedule coefficients:

```text
xt = sqrt(alpha_bar_t) × x0 + sqrt(1 - alpha_bar_t) × epsilon
```

At inference time, the model repeatedly denoises an initial random sample.

## Getting started

```bash
git clone https://github.com/Vansh-A1/DIFFUSION.git
cd DIFFUSION
python -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision matplotlib diffusers
```

Use a compatible PyTorch/Torchvision pair for your CPU or CUDA environment.

### Train the unconditional model

```bash
cd Unconditional_DDPM
python diffusion_scratch.py
```

MNIST is downloaded automatically. The script is configured for 50 epochs, 1,000 timesteps, a batch size of 128, and a learning rate of `2e-4`. It writes checkpoints and diagnostic image grids into `checkpoints/` and `samples/`.

### Train the conditional model

From the repository root:

```bash
cd conditional_DDMP
python train.py
```

This track uses 10 digit classes plus a null condition for classifier-free guidance. Its configuration sets 30 epochs, 1,000 timesteps, and 10% condition dropout.

Training can be slow on CPU. Start with fewer epochs or a smaller batch when exploring the code.

## Sampling and experiment notes

Each track includes an `inference.py` script. Before running it, inspect its checkpoint path and sampling configuration, and point it at a checkpoint produced by the matching training track. Model definitions, timestep settings, and class-conditioning settings must match that checkpoint.

The repository currently contains code rather than a published benchmark or bundled trained weights. A useful experiment report should include the training configuration, seed, hardware, loss curve, and generated sample grid.

## Next milestones

- [ ] Add a reproducible environment and unified configuration.
- [ ] Publish representative sample grids and training curves.
- [ ] Compare unconditional and conditional generation under controlled settings.
- [ ] Study guidance strength and sampling speed versus image quality.
- [ ] Explain the reverse sampling update alongside the code.

Built and maintained by [Vansh Joshi](https://github.com/Vansh-A1).

