# 🧠 Transformer from Scratch

## PyTorch Implementation for Machine Translation

This project is a complete **from-scratch implementation of the Transformer architecture** as introduced in  
["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), using only standard PyTorch modules.

🎯 **Goal**: Understand and implement the full Transformer model step by step — including custom tokenizer, data preprocessing, encoder-decoder model, and training loop — and apply it to a **machine translation task**.

---

## 🧑‍🏫 Based On

📺 [Implementing a Transformer from Scratch (with code)](https://www.youtube.com/watch?v=ISNdQcPhsts)  
👨‍💻 GitHub reference: [hkproj/pytorch-transformer](https://github.com/hkproj/pytorch-transformer)

> This project was developed as part of a hands-on learning journey.  
> I implemented all core components myself, closely following the video, and extended it with proper structure and modularity.

---

## ✅ Features Implemented

| Module | Description |
|--------|-------------|
| 🧩 Token Embedding | Embeds input token indices to vectors |
| 📍 Positional Encoding | Adds sinusoidal position info to embeddings |
| 🧠 Multi-Head Attention | Parallel scaled dot-product attention heads |
| 🏗 Feed Forward Layer | Two-layer dense network per token |
| 🔁 Residual Connection | With pre-norm LayerNormalization |
| 🧱 Encoder/Decoder | Stacked layers, causal/self/cross attention |
| 📤 Projection Layer | Maps decoder output to vocabulary space |
| ✂️ Word-level Tokenizer | Trained using HuggingFace `tokenizers` |
| 📚 Dataset Loader | Uses `opus_books` from HuggingFace |
| 🧪 Dataset Split | 90% train, 10% validation |
| ⚙️ Weight Init | Xavier initialization for all linear layers |

---

## 📦 Setup

```bash
pip install torch datasets tokenizers
```

## 🧪 Dataset: `opus_books`

- Bilingual sentence pairs for multiple language combinations.
- Loaded via HuggingFace `datasets` API.
- Trained custom word-level tokenizers (`WordLevel`) per language.
- Supports special tokens: `[UNK]`, `[PAD]`, `[SOS]`, `[EOS]`.

---

## 📁 Example Usage

```python
from dataset import get_ds
from transformer import build_transformer

# Load tokenizers and data
config = {
    "_src": "en",
    "lang_tgt": "fr",
    "tokenizer_file": "./tokenizers/tokenizer_{0}.json"
}

train_ds, val_ds, tokenizer_src, tokenizer_tgt = get_ds(config)

# Build transformer model
model = build_transformer(
    src_vocab_size=tokenizer_src.get_vocab_size(),
    tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
    src_seq_len=100,
    tgt_seq_len=100
)
```

## 🚧 Roadmap

- [x] Encoder & Decoder Modules  
- [x] Tokenizer Training & Serialization  
- [x] Dataset Loading & Splitting  
- [x] Custom `TranslationDataset` with tokenization + tensor conversion  
- [x] `DataLoader` with padding + batching  
- [ ] Full training loop (`CrossEntropyLoss` + teacher forcing)  
- [ ] Inference: `generate()` with greedy/beam search  
- [ ] Attention Map Visualization (inspired by Harvard NLP)

---

## 📚 References

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)  
- [hkproj/pytorch-transformer](https://github.com/hkproj/pytorch-transformer)  
- [YouTube: Implementing a Transformer from Scratch](https://www.youtube.com/watch?v=ISNdQcPhsts)  
- [HuggingFace Tokenizers](https://huggingface.co/docs/tokenizers/)  
- [HuggingFace Datasets](https://huggingface.co/docs/datasets)
