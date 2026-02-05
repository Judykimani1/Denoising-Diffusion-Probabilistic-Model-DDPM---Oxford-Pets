# Denoising Diffusion Probabilistic Model (DDPM) – Oxford Pets

This project implements a **Denoising Diffusion Probabilistic Model (DDPM)** using an **upgraded UNet architecture** with **residual blocks, skip connections, sinusoidal timestep embeddings, and self-attention**, trained on the **Oxford-IIIT Pets dataset**.

The goal of the project is to generate realistic **64×64 RGB images of pets** by learning to iteratively denoise Gaussian noise.

---

## 📌 Key Features

- Denoising Diffusion Probabilistic Model (DDPM)
- UNet backbone with encoder–decoder structure
- Residual blocks for stable deep training
- Skip connections to preserve spatial details
- Sinusoidal timestep embeddings
- Self-attention at intermediate resolutions
- Training checkpointing and resume support
- Visualization of generated samples and loss curves

---

## 📂 Dataset

**Oxford-IIIT Pets Dataset**

- RGB images of cats and dogs
- Images resized to **64×64**
- Pixel values normalized to **[-1, 1]**
- Used in an **unconditional** setting (labels ignored)

---

## 🧠 Model Architecture Overview

The model follows a **UNet-based architecture**, adapted for diffusion models.

### 🔹 High-level Structure

Input Image (x_t)
↓
Encoder (Downsampling path)
↓
Bottleneck
↓
Decoder (Upsampling path)
↓
Predicted Noise ε_θ(x_t, t)


---

## 🔹 UNet Encoder

- Consists of multiple **Residual Blocks**
- Spatial resolution decreases:  
  `64×64 → 32×32 → 16×16 → 8×8 → 4×4`
- Channel depth increases:  
  `3 → 128 → 256 → 512`
- **Self-attention layers** are applied at:
  - 32×32
  - 16×16
  - 8×8

---

## 🔹 Bottleneck

- Operates at the lowest spatial resolution (4×4)
- Contains:
  - Residual blocks
  - A self-attention layer
- Captures **global image structure** before reconstruction

---

## 🔹 UNet Decoder

- Upsamples feature maps back to the original resolution
- Uses **skip connections** from the encoder
- Skip connections help preserve:
  - Edges
  - Shapes
  - Fine spatial details

---

## 🔹 Residual Blocks

Each residual block:
- Applies normalization, activation, and convolution
- Adds the input back to the output (residual connection)
- Improves gradient flow and training stability
- Allows deeper networks without vanishing gradients

---

## 🔹 Self-Attention

Self-attention layers allow the model to:
- Capture **long-range dependencies**
- Relate distant parts of the image
- Improve global coherence (e.g., face structure, body shape)

This is especially useful for image generation where context matters.

---

## 🔹 Timestep Embedding

- Each diffusion step `t` is encoded using **sinusoidal embeddings**
- Embeddings are passed through an MLP
- Injected into every residual block
- Allows the network to learn **how much noise to remove at each timestep**


## 🔄 Diffusion Process

### Forward Process
Gradually adds Gaussian noise to images over `T` steps:
x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε


### Reverse Process
The model learns to predict the noise and remove it step by step:
x_t → x_{t-1} → ... → x_0
