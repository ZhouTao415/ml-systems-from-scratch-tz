# 🧠 ml-systems-from-scratch-tz

> Building state-of-the-art machine learning systems **from first principles**.  
> Fully modular PyTorch implementations of **Transformers**, **FlashAttention**, **Diffusion Models**, **Multimodal LLMs**, **RLHF**, and **Distributed Training** – all from scratch.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 What is this?

This repository is a **progressive implementation of modern ML systems**, rebuilt from scratch to better understand and teach:

- 🧠 Deep learning core building blocks  
- ⚡ High-performance techniques (FlashAttention, CUDA, Triton)  
- 📊 End-to-end training on real tasks (e.g. translation, image generation)

Inspired by [hkproj/pytorch-transformer](https://github.com/hkproj/pytorch-transformer) and based on video tutorials like [this one](https://www.youtube.com/watch?v=ISNdQcPhsts).

---

## 🗂️ Module Overview

| Module | 🔍 Focus | Topics | Status |
|--------|----------|--------|--------|
| [`01_transformer`](./01-transformer) | Machine Translation | Multi-head Attention, PE, LayerNorm, Encoder-Decoder | ✅ Complete |
| `02_flash_attention` | Efficient Attention | FlashAttention v2, Triton, CUDA kernels | 🚧 In progress |
| `03_stable_diffusion` | Generative Models | UNet, V-prediction, latent noise | 🧪 Planning |
| `04_multimodal_llm` | Vision-Language | ViT + LLM fusion, CLIP tokenization | 🧪 Planning |
| `05_rlhf_dpo` | Alignment | PPO, DPO, reward modeling | 🧪 Planning |

---

## 🧪 Example: Transformer for Translation

The [`01-transformer`](./01-transformer/) module includes:

- Full implementation of Vaswani et al. (2017)
- Trained from scratch on `opus_books` (HuggingFace datasets)
- Custom tokenizer via `tokenizers` library
- Modular design: encoder, decoder, projection, residuals
- 🚧 Training pipeline (WIP)

> 📓 Try it in [`01-transformer/demo.ipynb`](./01-transformer/demo.ipynb)

---

## 🚀 Quick Start

```bash
git clone https://github.com/taozhou/ml-systems-from-scratch-tz.git
cd ml-systems-from-scratch-tz
conda env create -f environment.yml
conda activate msfs

# Run Transformer training (once ready)
python 01-transformer/train.py --config configs/base.yaml
