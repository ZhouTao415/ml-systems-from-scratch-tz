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

        # encoder_mask:
        # we are increasing the size of the encoder input sentence by adding padding tokens,
        # but we dont want to these padding tokens to participate in the self-attention
        # so we build a mask that these padding tokens will not be senn by the self-attention
        # how can we build this mask?
        # all token are not padding toekens are ok, padding tokens are not ok
        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()  # (1, 1, seq_len)
        # causal mask for the decoder input:
        #  we want to the word can only look at the previous words 
        #  and each word can only look at non-padding words
        # we only want the real words to participate in the self-attention
        # we also dont want to each word to watch at wirds that come after it
        # so onlu that word comes before it can be seen
        
        # Decoder mask:
        # 1. Causal mask to prevent attending to future positions
        # 2. Ignore [PAD] tokens
        decoder_pad_mask = (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() # (1, seq_len)
        decoder_mask = decoder_pad_mask & causal_mask(decoder_input.size(0)) # (1, seq_len) & (1, seq_len, seq_len)
        return {
            'encoder_input': encoder_input, # (seq_len)
            'decoder_input': decoder_input, # (seq_len)
            'encoder_mask': encoder_mask, # (1, 1, seq_len) ~ (batch_size, num_heads, seq_len)
            'decoder_mask': decoder_mask,
            'label': label, # (seq_len)
            'src_text': src_text, 
            'tgt_text': tgt_text  
        }
    
def causal_mask(size):
    # this returen all the values above the diagonal and everything below the diagonal is 0
    # but we want the opposite, so we invert it
    mask = torch.tril(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0