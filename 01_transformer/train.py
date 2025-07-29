from dataset import BilingualDataset, causal_mask
from model import build_transformer

from config import get_config, get_weights_file_path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# We use the library from huggingface
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter

import warnings
from tqdm import tqdm
from pathlib import Path

def get_all_sentences(dataset, lang):
    """
    Generator that yields all sentences in a specific language from a dataset.

    Args:
        dataset: Huggingface dataset with 'translation' field.
        lang (str): Language code (e.g., 'en', 'de').
        # item['translation'] is a dictionary with language codes as keys
        # e.g., {'en': 'Hello', 'fr': 'Bonjour'}

    Yields:
        str: Sentence in the specified language.
    """
    for item in dataset:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    """
    Load an existing tokenizer from file or train a new one and save it.

    Args:
        config (dict): Configuration dictionary with tokenizer path format.
        dataset: Huggingface dataset to train the tokenizer.
        lang (str): Language code (e.g., 'en', 'de').

    Returns:
        Tokenizer: Trained or loaded Huggingface tokenizer.
    """
    # Create the abs path giving the relative path
    # lang = 'en'
    # e.g. onfig['tokenizer_file'] = '../tokenizers/tokenizer_{0}.json'
    # '../tokenizers/tokenizer_en.json'
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Create a new word-level tokenizer with [UNK] for unknown tokens
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()

        # Train the tokenizer on the dataset
        # UNK = unknown token
        # PAD = padding token used to train the transformer
        # SOS = start of sequence token
        # EOS = end of sequence token
        # Train tokenizer with special tokens and frequency threshold
        trainer = WordLevelTrainer(
            special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'],
            min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)

        # Save to disk
        tokenizer.save(str(tokenizer_path))
    else:
        # Load existing tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_ds(config):
    """
    Loads the dataset and prepares tokenizers and dataloaders for training and validation.

    Args:
        config (dict): Configuration dictionary. Expected keys:
            - lange_src (str): Source language code (e.g., 'en')
            - lange_tgt (str): Target language code (e.g., 'de')
            - tokenizer_file (str): Path template to save/load tokenizers
            - seq_len (int): Fixed input sequence length for model
            - batch_size (int): Batch size for training

    Returns:
        Tuple:
            - train_data_loader (DataLoader): Training set DataLoader
            - val_data_loader (DataLoader): Validation set DataLoader
            - tokenizer_src (Tokenizer): Source language tokenizer
            - tokenizer_tgt (Tokenizer): Target language tokenizer
    """
    # Load translation dataset from HuggingFace Datasets
    ds_name = 'opus_books'
    lang_pair = f"{config['lange_src']}-{config['lange_tgt']}"
    ds_raw = load_dataset(ds_name, lang_pair, split='train')

    # Build or load tokenizers for both source and target languages
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lange_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lange_tgt'])

    # Split dataset: 90% training, 10% validation
    train_size = int(0.9 * len(ds_raw))
    val_size = len(ds_raw) - train_size
    train_raw, val_raw = random_split(ds_raw, [train_size, val_size])

    # Wrap with custom Dataset class to handle tokenization and tensor preparation
    train_ds = BilingualDataset(
        ds=train_raw,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        src_lang=config['lange_src'],
        tgt_lang=config['lange_tgt'],
        seq_len=config['seq_len']
    )
    val_ds = BilingualDataset(
        ds=val_raw,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        src_lang=config['lange_src'],
        tgt_lang=config['lange_tgt'],
        seq_len=config['seq_len']
    )

    # Compute max sequence length in raw dataset (for information/debugging)
    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lange_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lange_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    print(f"Max source length: {max_len_src}, Max target length: {max_len_tgt}")

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_loader, val_loader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_size_src: int, vocab_size_tgt: int):
    """
    Builds and returns a Transformer model using the provided configuration and vocabulary sizes.

    Args:
        config (dict): Configuration dictionary with model parameters.
        vocab_size_src (int): Vocabulary size of the source language.
        vocab_size_tgt (int): Vocabulary size of the target language.

    Returns:
        nn.Module: An instance of the Transformer model.
    """
    model = build_transformer(
        vocab_size_src=vocab_size_src,
        vocab_size_tgt=vocab_size_tgt,
        src_seq_len=config['seq_len'],
        tgt_seq_len=config['seq_len'],
        d_model=config['d_model']
    )
    return model

def train_model(config):
    """Main training loop with checkpointing and logging."""

    # Setup: device, model folder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # Data & model
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    model = model.to(device)

    # TensorBoard logger
    writer = SummaryWriter(config['experiment_name'])

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # Resume from checkpoint if specified
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model from {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']

    # Loss function (with label smoothing), ignore [PAD] token
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id('[PAD]'),
        label_smoothing=0.1
    ).to(device)

    # === Training Loop ===
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc= f'Processing epoch {(epoch + 1):02d}/{config["num_epochs"]}')

        for batch in batch_iterator:
            # Move batch data to device
            encoder_input = batch['encoder_input'].to(device) # （batch_size, seq_len)
            decoder_input = batch['decoder_input'].to(device) #  (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch_size, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch_size, 1, seq_len, seq_len)

            # Run the tensor through the transformer model
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) #   (batch_size, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch_size, seq_len, tgt_vocab_size)

            # label: each dim tell us what is the position of the token in the vocabulary
            label = batch['label'].to(device) #  [(batch_size, seq_len)]

            # [batch_size, seq_len, tgt_vocab_size] -> [batch_size * seq_len, tgt_vocab_size]
            # Flatten for loss computation
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()),
                label.view(-1)
            )
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"}) # this show the loss in the progress bar

            # Log loss to TensorBoard
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

           # === Backward & Optimize ===
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #  the global step is used for tensorbaord to keep track of the loss
            global_step += 1

        # === Save checkpoint after each epoch ===
        model_filename = get_weights_file_path(config, f'{epoch + 1}:02d')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
    print("✅ Training complete. Check the 'weights' folder for saved checkpoints.")

# Entry point
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
