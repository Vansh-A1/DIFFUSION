# DDPM From Scratch

A PyTorch implementation of a **Denoising Diffusion Probabilistic Model (DDPM) built from scratch**.

The main purpose of this project is to understand how diffusion models work internally by implementing the core components without using high-level diffusion libraries such as Hugging Face Diffusers.

The model is currently trained on the **MNIST dataset** and learns to predict the Gaussian noise added to images during the forward diffusion process.

---

## 🎯 Project Goal

The goal of this project is not to build the most optimized diffusion model.

Instead, it is to understand the complete DDPM pipeline:

```text
Original Image x₀
       │
       ▼
Forward Diffusion
       │
       ▼
Noisy Image xₜ
       │
       │ timestep t
       ▼
      U-Net
       │
       ▼
Predicted Noise εθ(xₜ,t)
       │
       ▼
MSE Loss
       │
       ▼
Model learns to predict noise
