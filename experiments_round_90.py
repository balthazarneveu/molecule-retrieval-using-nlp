from properties import (
    NB_EPOCHS, BATCH_SIZE, LEARNING_RATE, TOKENIZER_NAME, NAME, ANNOTATIONS, WEIGHT_DECAY, OPTIMIZER,
    DISTILBERT, BIG_GCN,
    PLATEAU
)
from pathlib import Path
from multimodal_model import MultimodalModel
from language_model import TextEncoder
from graph_model import BasicGraphEncoder
from experiments_generic import generic_experiment


def get_round_90_experience(exp: int, configuration: dict, root_dir: Path = None, backup_root: Path = None):
    """Lea's experiments"""
    assert exp >= 9000 and exp < 9099, f"Experiment {exp} is not in the round 90"
    if exp == 9000:
        configuration[BATCH_SIZE] = (32, 32, 32)
        configuration[NB_EPOCHS] = 60
        configuration[OPTIMIZER][LEARNING_RATE] = 5e-5
        configuration[OPTIMIZER][WEIGHT_DECAY] = 1.
        configuration[NAME] = 'FBERT-GCN'
        configuration[ANNOTATIONS] = 'Frozen BERT - Trainable GCN'
        graph_encoder = BasicGraphEncoder(num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300)
        text_encoder = TextEncoder(configuration[TOKENIZER_NAME], freeze=True)
        model = MultimodalModel(graph_encoder, text_encoder)
        configuration[ANNOTATIONS] += "- LEA"
    elif exp == 9001:
        model, configuration = generic_experiment(
            configuration,
            llm=DISTILBERT, graph=BIG_GCN,
            n=200,
            b=128, lr=3e-4, wd=1e-1,
            scheduler=PLATEAU, scheduler_configuration=dict(patience=10, factor=0.8),
            lora=False, quantization=None
        )
    elif exp == 9002:
        model, configuration = generic_experiment(
            configuration,
            llm=DISTILBERT, graph=FAT_GCN,
            n=200,
            b=128, lr=3e-4, wd=1e-1,
            scheduler=PLATEAU, scheduler_configuration=dict(patience=10, factor=0.8),
            lora=False, quantization=None
        )
    elif exp == 9003:
        model, configuration = generic_experiment(
            configuration,
            llm=SCIBERT, graph=FAT_GCN,
            n=200,
            b=128, lr=1e-4, wd=1e-1,
            scheduler=PLATEAU, scheduler_configuration=dict(patience=10, factor=0.8),
            lora=False, quantization=None
        )
    print(configuration)
    return model, configuration
