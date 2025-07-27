import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# We use the library from huggingface
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

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
    Load dataset and build tokenizers for source and target languages.

    Args:
        config (dict): Configuration dictionary with language and tokenizer paths.

    Returns:
        Tuple containing:
            - train_ds_raw: 90% of raw dataset
            - val_ds_raw: 10% of raw dataset
            - tokenizer_src: Tokenizer for source language
            - tokenizer_tgt: Tokenizer for target language
    """
    ds_name = 'opus_books'
    lang_pair = f"{config['lange_src']}-{config['lange_tgt']}"
    ds_raw = load_dataset(ds_name, lang_pair, split='train')
    
    # Build Tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lange_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lange_tgt'])

    # Keep 90% of the dataset for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    
    # Create a custom dataset class
    """
    The data set that our model will use to access the tensor directly
    because we just create the tokenizer, and just loaded the dataset
    we need to create the tensor that our model will use
    """