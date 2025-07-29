from pathlib import Path

def get_config():
    """
    Returns the training configuration as a dictionary.

    Keys include:
        - batch_size (int): Training batch size
        - num_epochs (int): Total training epochs
        - lr (float): Learning rate
        - seq_len (int): Fixed input sequence length
        - d_model (int): Model hidden dimension
        - lang_src (str): Source language code (e.g., 'en')
        - lang_tgt (str): Target language code (e.g., 'it')
        - model_folder (str): Folder to save model weights
        - model_filename (str): Prefix for model weight files
        - preload (str or None): Optional epoch string to resume from
        - tokenizer_file (str): Path template for saving tokenizer (with {0} as language code)
        - experiment_name (str): Path for logging (e.g., TensorBoard)

    Returns:
        dict: Training configuration
    """
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 1e-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_filename": "transformer_model_",
        "preload": None,  # e.g., '10' to resume from epoch 10 (transformer_model_10.pt)
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/transformer_model"
    }


def get_weights_file_path(config, epoch: str) -> str:
    """
    Constructs the file path for saving/loading model weights.

    Args:
        config (dict): Configuration dictionary
        epoch (str): Epoch number (e.g., '10')

    Returns:
        str: Full path to the model checkpoint file
    """
    model_folder = config['model_folder']
    model_basename = config['model_filename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)
