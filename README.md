# ml-systems-from-scratch-tz

> Implementing state‑of‑the‑art deep‑learning systems **from first principles**.  
> **Transformers • FlashAttention • Multimodal LLMs • Stable Diffusion • RLHF • Quantisation & Distributed Training**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 本仓库为个人学习笔记，面向中英双语读者。敬请斧正。

---

## ✨ Highlights

| Module | Lines of Code | Key Topics | Notebook / Video |
|--------|---------------|-----------|------------------|
| **01‑transformer** | 480 | KV‑cache • Rotary PE | [notebook](./01-transformer/demo.ipynb) |
| **02‑flash‑attention** | 550 | Triton kernels • CUDA | [notebook](./02-flash-attention) |
| **03‑stable‑diffusion** | 680 | UNet • V‑prediction | … |
| **04‑multimodal‑llm** | 620 | ViT + LLM fusion | … |
| **05‑rlhf‑dpo** | 400 | PPO • DPO • reward models | … |

---

## 🚀 Quick Start

```bash
git clone https://github.com/taozhou/ml-systems-from-scratch-tz.git
cd ml-systems-from-scratch-tz
conda env create -f environment.yml
conda activate msfs
python 01-transformer/train.py --config configs/base.yaml
```
## 🙏 Credits

Major parts of this repository are **faithful re‑implementations** of the excellent code and lectures by  
[Umar Jamil (@hkproj)](https://github.com/hkproj).  
Original source projects are MIT‑licensed; see their repos for details.

