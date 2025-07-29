import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    """
    Custom PyTorch Dataset for bilingual translation data.
    Prepares encoder and decoder input sequences with appropriate special tokens and padding.
    # url: https://huggingface.co/datasets/Helsinki-NLP/opus_books
    """

    def __init__(
        self,
        ds,
        tokenizer_src,
        tokenizer_tgt,
        src_lang,
        tgt_lang,
        seq_len
    ) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # Convert special tokens to input IDs
        # torch.int64 is required because vocabulary indices can exceed 32-bit range
        self.sos_token = torch.tensor(
            [tokenizer_src.token_to_id('[SOS]')],
            dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [tokenizer_src.token_to_id('[EOS]')],
            dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [tokenizer_src.token_to_id('[PAD]')],
            dtype=torch.int64
        )

    def __len__(self):
        """Returns the total number of examples in the dataset."""
        return len(self.ds)
    
    def __getitem__(self, index: any) -> any:
        """
        Processes a single sample:
        - Tokenizes source and target text
        - Adds [SOS], [EOS], [PAD] tokens
        - Returns encoder/decoder inputs, masks

        Returns:
            A dictionary with:
                - encoder_input: Tensor of shape (seq_len,)
                - decoder_input: Tensor of shape (seq_len,)
                - encoder_mask: Tensor of shape (1, 1, seq_len)
                - decoder_mask: Tensor of shape (1, seq_len, seq_len)
        """
        src_tgt_pair = self.ds[index]
        src_text = src_tgt_pair['translation'][self.src_lang]
        tgt_text = src_tgt_pair['translation'][self.tgt_lang]

        # Tokenize the input sentences:
        # - Each sentence is split into tokens (typically by whitespace or punctuation)
        # - Each token is then mapped to its corresponding vocabulary ID
        # - The result is a list of integer token IDs
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Pad sequences to fixed length:
        # The model expects all inputs to have the same fixed sequence length (self.seq_len).
        # Since most sentences are shorter, we pad them with [PAD] tokens.

        # For the encoder: we reserve space for [SOS] and [EOS] tokens, so we subtract 2.
        # For the decoder input: only [SOS] is added (the label will get [EOS]), so subtract 1.
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Input sentence is too long for the configured sequence length.")
        
        # Add [SOS] and [EOS] tokens to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Add [SOS] token to the target text
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # The label / target text is the decoder input shifted by one position
        # Add [EOS] tp the label (what we expect as ouput from the decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Check that all sequences are correctly padded
        assert encoder_input.size(0) == self.seq_len, "Encoder input size mismatch"
        assert decoder_input.size(0) == self.seq_len, "Decoder input size mismatch"
        assert label.size(0) == self.seq_len, "Label size mismatch"

        # === Encoder Mask ===
        # Prevents attention from attending to [PAD] tokens in the encoder input.
        # Shape: (1, 1, seq_len) — suitable for broadcasting over (batch_size, num_heads, seq_len)
        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()

        # === Decoder Mask ===
        # Combines two masks:
        # 1. Padding mask: ignores [PAD] tokens in decoder input.
        # 2. Causal mask: ensures that each token can only attend to itself and previous tokens (no future tokens).
        decoder_pad_mask = (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()  # (1, 1, seq_len)
        decoder_mask = decoder_pad_mask & causal_mask(decoder_input.size(0))  # (1, seq_len, seq_len)

        return {
            'encoder_input': encoder_input,    # (seq_len,)
            'decoder_input': decoder_input,    # (seq_len,)
            'encoder_mask': encoder_mask,      # (1, 1, seq_len)
            'decoder_mask': decoder_mask,      # (1, seq_len, seq_len)
            'label': label,                    # (seq_len,)
            'src_text': src_text,              # Original source sentence
            'tgt_text': tgt_text               # Original target sentence
        }

def causal_mask(size: int) -> torch.Tensor:
    """
    Generates a lower-triangular (causal) mask for decoder self-attention.
    Ensures position i can only attend to positions <= i.

    Returns:
        Tensor of shape (1, size, size), where True indicates allowed attention.
    """
    mask = torch.tril(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0