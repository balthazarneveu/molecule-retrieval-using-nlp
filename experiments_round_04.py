from properties import (
    NB_EPOCHS, BATCH_SIZE, LEARNING_RATE, TOKENIZER_NAME, NAME, ANNOTATIONS, WEIGHT_DECAY, OPTIMIZER,
)
from pathlib import Path
from multimodal_model import MultimodalModel
from language_model import TextEncoder
from graph_model import BasicGraphEncoder


def get_round_4_experience(exp: int, configuration: dict, root_dir: Path = None, backup_root: Path = None):
    """Scientific language model

    https://github.com/balthazarneveu/molecule-retrieval-using-nlp/issues/16
    """
    assert exp >= 400 and exp <= 499, "round 4 between 400 and 499"
    if exp == 400:
        configuration[BATCH_SIZE] = (32, 32, 32)    # RTX2060
        configuration[NB_EPOCHS] = 150
        configuration[OPTIMIZER][LEARNING_RATE] = 7e-6
        configuration[OPTIMIZER][WEIGHT_DECAY] = 0.1
        configuration[NAME] = 'SciBERT-GCN'
        configuration[ANNOTATIONS] = 'Trainable SciBERT - base GCN'
        configuration[TOKENIZER_NAME] = "allenai/scibert_scivocab_uncased"
        graph_encoder = BasicGraphEncoder(num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300)
        text_encoder = TextEncoder(configuration[TOKENIZER_NAME], freeze=False)
        model = MultimodalModel(graph_encoder, text_encoder)
        return model, configuration
