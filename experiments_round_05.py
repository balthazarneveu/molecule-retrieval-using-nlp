from properties import (
    NB_EPOCHS, BATCH_SIZE, LEARNING_RATE, TOKENIZER_NAME, NAME, ANNOTATIONS, WEIGHT_DECAY, OPTIMIZER,
    SCHEDULER_CONFIGURATION, SCHEDULER
)
from pathlib import Path
from multimodal_model import MultimodalModel
from language_model import TextEncoder
from graph_model import BasicGraphEncoder, BigGraphEncoder, FatGraphEncoder

import torch
from typing import Tuple
from lora import get_lora_configuration, get_quantization_configuration


SHORT_NAMES = ["distilbert", "scibert", "bert"]


def lora_exp(
    configuration: dict,
    b: int = 32,
    n: int = 150,
    lr: float = 7e-6,
    wd: float = 0.,
    model_name: str = "distilbert",
    graph_encoder=None,
    quantization=None
) -> Tuple[torch.nn.Module, dict]:
    configuration[NB_EPOCHS] = n
    configuration[OPTIMIZER][LEARNING_RATE] = lr
    configuration[OPTIMIZER][WEIGHT_DECAY] = wd
    configuration[TOKENIZER_NAME] = model_name
    configuration[BATCH_SIZE] = (b, b, b)
    assert model_name in SHORT_NAMES, f"{model_name} must be in {SHORT_NAMES}"
    if model_name == "distilbert":
        configuration[TOKENIZER_NAME] = "distilbert-base-uncased"
        configuration[NAME] = 'LoraBERT-GCN'
        configuration[ANNOTATIONS] = 'Trainable Lora Distil BERT'
    if model_name == "scibert":
        configuration[TOKENIZER_NAME] = "allenai/scibert_scivocab_uncased"
        configuration[NAME] = 'LoraSciBERT-GCN'
        configuration[ANNOTATIONS] = 'Lora SciBERT'
    lora_dict = get_lora_configuration(configuration[TOKENIZER_NAME])
    if graph_encoder is None:
        graph_encoder = BasicGraphEncoder(num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300)
        configuration[ANNOTATIONS] += "- base GCN"
    q_dict = None
    if quantization is not None:
        if quantization == "nf4":
            q_dict = get_quantization_configuration()
            configuration[NAME] = configuration[NAME].replace("Lora", "QLora")
        else:
            raise NameError(f"Quantization {quantization} not implemented")
    text_encoder = TextEncoder(configuration[TOKENIZER_NAME], freeze=False, lora=lora_dict, quantization_config=q_dict)
    model = MultimodalModel(graph_encoder, text_encoder)

    return model, configuration


def get_round_5_experience(exp: int, conf: dict, root_dir: Path = None, backup_root: Path = None):
    """Use LoRA to drastically reduce the number of parameters of the model

    https://github.com/balthazarneveu/molecule-retrieval-using-nlp/issues/19

    Note
    ====
    - batch_size = 16  # T500
    - batch_size = 32  # RTX2060
    """
    # After 20 epochs

    # 500 LoraBERT-GCN	          17.8% <<< LR 7e-6
    # 501 LoraSciBERT-GCN	      27.6% <<< LR 7e-6
    # 504 LoraBERT-GCN            38.9% <<< LR 5e-5
    # 505 LoraBERT-GCN	          47.6% *** LR 1e-4
    # 506 LoraBERT-GCN	          32.1% <<< LR 1e-3
    # 507 LoraSciBERT-biggerGCN   47.7% <<< LR 3e-5  Kaggle

    # 068 Base-HP-search	      46.1%
    assert exp >= 500 and exp <= 599, "round 5 between 500 and 599"
    if exp == 500:
        model, conf = lora_exp(conf, b=32, n=150, lr=7e-6, wd=0.1, model_name="distilbert")
    elif exp == 501:
        model, conf = lora_exp(conf, b=32, n=60, lr=7e-6, wd=0.1, model_name="scibert")
    elif exp == 502:
        model, conf = lora_exp(conf, b=256, n=200, lr=3e-4, wd=0.1, model_name="scibert")  # probably A5000 required
    elif exp == 503:
        graph_encoder = BigGraphEncoder(num_node_features=300, nout=768, nhid=256, graph_hidden_channels=512)
        model, conf = lora_exp(conf, b=128, n=150, lr=7e-6, wd=0.1, model_name="scibert", graph_encoder=graph_encoder)
        conf["GCN-architecture"] = {
            "depth": 5,
            "GCN-FC-hidden-size": 512,
            "GCN-hidden-size": 256,
            "GNN-out-size": 768,
        }
        conf[ANNOTATIONS] += "- bigger GCN"
        conf[NAME] = conf[NAME].replace("GCN", "biggerGCN")
    elif exp == 504:
        # LR 5e-5 does is not optimal
        model, conf = lora_exp(conf, b=32, n=20, lr=5e-5, wd=0.1, model_name="distilbert")
    elif exp == 505:  # *************** #
        # LR 1e-4 seems well balanced
        model, conf = lora_exp(conf, b=32, n=20, lr=1e-4, wd=0.1, model_name="distilbert")
    elif exp == 506:
        # LR 1e-3 is too high
        model, conf = lora_exp(conf, b=32, n=20, lr=1e-3, wd=0.1, model_name="distilbert")
    elif exp in [507, 511, 512, 515, 517, 518, 571]:
        # Seems to have a good convergence
        # begining looks as good as training all BERT parameters with mega batch size 128 - exp 68
        if exp == 507:
            lr = 3e-5
            n = 40
            b = 32
        elif exp == 511:
            lr = 3e-4
            n = 10
            b = 32
        elif exp == 512:
            lr = 3e-4
            n = 40
            b = 48
        elif exp == 515:
            lr = 1e-4
            n = 45
            b = 48
        elif exp == 517:
            lr = 1e-4
            n = 45
            b = 48
            conf[SCHEDULER] = "ReduceLROnPlateau"
            conf[SCHEDULER_CONFIGURATION] = dict(patience=5, factor=0.5)
        elif exp == 518:
            lr = 1e-4
            n = 200
            b = 48
            conf[SCHEDULER] = "ReduceLROnPlateau"
            conf[SCHEDULER_CONFIGURATION] = dict(patience=5, factor=0.5)
        elif exp == 571:
            lr = 1e-4
            n = 150
            b = 64
            conf[SCHEDULER] = "ReduceLROnPlateau"
            conf[SCHEDULER_CONFIGURATION] = dict(patience=5, factor=0.5)
        graph_encoder = BigGraphEncoder(num_node_features=300, nout=768, nhid=256, graph_hidden_channels=512)
        model, conf = lora_exp(conf, b=b, n=n, lr=lr, wd=0.1, model_name="scibert", graph_encoder=graph_encoder)
        conf["GCN-architecture"] = {
            "depth": 5,
            "GCN-FC-hidden-size": 512,
            "GCN-hidden-size": 256,
            "GNN-out-size": 768,
        }
        conf[ANNOTATIONS] += "- bigger GCN"
        conf[NAME] = conf[NAME].replace("GCN", "biggerGCN")
    # HP Fast start grid search LR
    elif exp == 508:  # **BEST START SO FAR**
        model, conf = lora_exp(conf, b=32, n=10, lr=3e-4, wd=0.1, model_name="distilbert")
    elif exp == 509:
        model, conf = lora_exp(conf, b=32, n=10, lr=6e-4, wd=0.1, model_name="distilbert")
    elif exp == 510:
        model, conf = lora_exp(conf, b=32, n=10, lr=1e-4, wd=0., model_name="distilbert")
    elif exp == 513:  # like 508 but more epochs
        model, conf = lora_exp(conf, b=32, n=150, lr=3e-4, wd=0.1, model_name="distilbert")
    elif exp == 514:
        model, conf = lora_exp(conf, b=128, n=200, lr=3e-4, wd=0.1, model_name="distilbert")  # Lea
    elif exp == 516:  # like 508 but more epochs and with a LR scheduler
        model, conf = lora_exp(conf, b=32, n=150, lr=3e-4, wd=0.1, model_name="distilbert")
        conf[SCHEDULER] = "ReduceLROnPlateau"
        conf[SCHEDULER_CONFIGURATION] = dict(patience=5, factor=0.5)
    elif exp == 519:  # Kaggle experiment
        model, conf = lora_exp(conf, b=128, n=150, lr=3e-4, wd=0.1, model_name="distilbert", quantization="nf4")
        conf[SCHEDULER] = "ReduceLROnPlateau"
        conf[SCHEDULER_CONFIGURATION] = dict(patience=5, factor=0.5)
    elif exp == 520:  # Not a bigger GCN
        model, conf = lora_exp(conf, b=32, n=150, lr=3e-4, wd=0.1, model_name="distilbert")
        conf[SCHEDULER] = "ReduceLROnPlateau"
        conf[SCHEDULER_CONFIGURATION] = dict(patience=5, factor=0.5)
    elif exp == 521:
        graph_encoder = BigGraphEncoder(num_node_features=300, nout=768, nhid=256, graph_hidden_channels=512)
        model, conf = lora_exp(conf, b=32, n=150, lr=3e-4, wd=0.1, model_name="distilbert", graph_encoder=graph_encoder)
        conf[SCHEDULER] = "ReduceLROnPlateau"
        conf[SCHEDULER_CONFIGURATION] = dict(patience=5, factor=0.5)
        conf[ANNOTATIONS] += "- bigger GCN"
        conf[NAME] = conf[NAME].replace("GCN", "biggerGCN")
    elif exp == 522:  # Kaggle experiment - change scheduler
        model, conf = lora_exp(conf, b=128, n=75, lr=3e-4, wd=0.1, model_name="distilbert", quantization="nf4")
        conf[SCHEDULER] = "ReduceLROnPlateau"
        conf[SCHEDULER_CONFIGURATION] = dict(patience=6, factor=0.8)
    elif exp == 523:
        graph_encoder = FatGraphEncoder(num_node_features=300, nout=768, nhid=256, graph_hidden_channels=512)
        model, conf = lora_exp(conf, b=32, n=75, lr=3e-4, wd=0.1, model_name="distilbert", graph_encoder=graph_encoder)
        conf[SCHEDULER] = "ReduceLROnPlateau"
        conf[SCHEDULER_CONFIGURATION] = dict(patience=5, factor=0.5)
        conf[ANNOTATIONS] += "- fat GCN"
        conf[NAME] = conf[NAME].replace("GCN", "FatGCN")
    elif exp == 524:
        graph_encoder = BigGraphEncoder(num_node_features=300, nout=768, nhid=256, graph_hidden_channels=512)
        model, conf = lora_exp(conf, b=32, n=150, lr=5e-4, wd=0.1, model_name="distilbert", graph_encoder=graph_encoder)
        conf[SCHEDULER] = "CosineAnnealingWarmRestarts"
        conf[SCHEDULER_CONFIGURATION] = dict(T_0=30, T_mult=1, eta_min=1e-5)
        conf[ANNOTATIONS] += "- bigger GCN"
        conf[NAME] = conf[NAME].replace("GCN", "biggerGCN")
    elif exp == 525:
        graph_encoder = BigGraphEncoder(num_node_features=300, nout=768, nhid=256, graph_hidden_channels=512)
        model, conf = lora_exp(conf, b=32, n=150, lr=1e-3, wd=0.1, model_name="distilbert", graph_encoder=graph_encoder)
        conf[SCHEDULER] = "CosineAnnealingWarmRestarts"
        conf[SCHEDULER_CONFIGURATION] = dict(T_0=40, T_mult=2, eta_min=1e-5)
        conf[ANNOTATIONS] += "- bigger GCN"
        conf[NAME] = conf[NAME].replace("GCN", "biggerGCN")
    elif exp == 570:  # exp 521 bigger batch size
        graph_encoder = BigGraphEncoder(num_node_features=300, nout=768, nhid=256, graph_hidden_channels=512)
        model, conf = lora_exp(conf, b=64, n=150, lr=3e-4, wd=0.1, model_name="distilbert", graph_encoder=graph_encoder)
        conf[SCHEDULER] = "ReduceLROnPlateau"
        conf[SCHEDULER_CONFIGURATION] = dict(patience=5, factor=0.5)
        conf[ANNOTATIONS] += "- bigger GCN"
        conf[NAME] = conf[NAME].replace("GCN", "biggerGCN")
    else:
        raise NameError(f"Experiment {exp} not implemented")

    return model, conf
