from properties import (
    NB_EPOCHS, BATCH_SIZE, LEARNING_RATE, TOKENIZER_NAME, NAME, ANNOTATIONS, WEIGHT_DECAY, OPTIMIZER,
    MAX_STEP_PER_EPOCH, BETAS
)
from pathlib import Path
from multimodal_model import MultimodalModel
from language_model import TextEncoder, Adapter
from graph_model import BigGraphEncoder


def get_round_2_experience(exp: int, configuration: dict, root_dir: Path = None, backup_root: Path = None):
    """Bigger GCN
    https://github.com/balthazarneveu/molecule-retrieval-using-nlp/issues/13
    """
    assert exp >= 200 and exp <= 299, "round 2 between 200 and 299"
    if exp == 200:
        configuration[BATCH_SIZE] = (32, 32, 32)    # RTX2060
        configuration[NB_EPOCHS] = 60
        configuration[OPTIMIZER][LEARNING_RATE] = 5e-6
        configuration[OPTIMIZER][WEIGHT_DECAY] = 0.1
        configuration[NAME] = 'BERT-bGCN'
        configuration[ANNOTATIONS] = 'Trainable BERT - Bigger GCN D=5'
        configuration["GCN-architecture"] = {
            "depth": 5,
            "GCN-FC-hidden-size": 512,
            "GCN-hidden-size": 256,
            "GNN-out-size": 768,
        }
        graph_encoder = BigGraphEncoder(num_node_features=300, nout=768, nhid=256, graph_hidden_channels=512)
        text_encoder = TextEncoder(configuration[TOKENIZER_NAME], freeze=False)
        model = MultimodalModel(graph_encoder, text_encoder)
        return model, configuration
