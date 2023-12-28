from properties import (
    NB_EPOCHS, BATCH_SIZE, LEARNING_RATE, TOKENIZER_NAME, NAME, ANNOTATIONS, WEIGHT_DECAY, OPTIMIZER,
)
from pathlib import Path
from multimodal_model import MultimodalModel
from language_model import TextEncoder
from graph_model import BasicGraphEncoder


def get_round_80_experience(exp: int, configuration: dict, root_dir: Path = None, backup_root: Path = None):
    """Basile's experiments"""
    assert exp >= 8000 and exp <= 8999, "baseline exp must be <= 8999"
    if exp == 8000:
        configuration[BATCH_SIZE] = (32, 32, 32)
        configuration[NB_EPOCHS] = 60
        configuration[OPTIMIZER][LEARNING_RATE] = 5e-5
        configuration[OPTIMIZER][WEIGHT_DECAY] = 1.
        configuration[NAME] = 'FBERT-GCN'
        configuration[ANNOTATIONS] = 'Frozen BERT - Trainable GCN'
        graph_encoder = BasicGraphEncoder(num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300)
        text_encoder = TextEncoder(configuration[TOKENIZER_NAME], freeze=True)
        model = MultimodalModel(graph_encoder, text_encoder)
    if exp == 8001:
        configuration[BATCH_SIZE] = (32, 32, 32)
        configuration[NB_EPOCHS] = 2
        configuration[OPTIMIZER][LEARNING_RATE] = 5e-5
        configuration[OPTIMIZER][WEIGHT_DECAY] = 1.
        configuration[NAME] = 'FBERT-GCN'
        configuration[ANNOTATIONS] = 'Frozen BERT - Trainable GCN'
        graph_encoder = BasicGraphEncoder(num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300)
        text_encoder = TextEncoder(configuration[TOKENIZER_NAME], freeze=True)
        model = MultimodalModel(graph_encoder, text_encoder)
    configuration[NAME] += "- BASILE"
    return model, configuration