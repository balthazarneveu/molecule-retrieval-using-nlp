from properties import (
    SCHEDULER, SCHEDULER_CONFIGURATION,
    DISTILBERT, SCIBERT, BIG_GCN, FAT_GCN, PLATEAU,
    LOSS, NAME, BATCH_SIZE, LOSS_TEMPERED_CROSSENTROPY, LOSS_BINARY_CROSSENTROPY
)
from pathlib import Path
from properties import OUT_DIR
from experiments_generic import generic_experiment, custom_lr
from graph_model import FatGraphEncoder, UltraFatGraphEncoder
from functools import partial
import torch
from huggingface_hub import hf_hub_download


def get_round_8_experience(exp: int, configuration: dict, root_dir: Path = None, backup_root: Path = None):
    """"""
    assert exp >= 800 and exp < 899, f"Experiment {exp} is not in the round 8"
    if exp == 800:
        # Experiment with big batch size, tempered loss + BCE.
        lr = 3e-4
        batch_size = 256
        model, configuration = generic_experiment(
            configuration,
            llm=DISTILBERT,
            graph=BIG_GCN,
            n=400,
            b=batch_size,
            lr=lr,
            wd=1e-1,
            lora=False,
            quantization=None,
            mixed_precision=True,
            temperature=True
        )
        configuration[LOSS] = ",".join([LOSS_TEMPERED_CROSSENTROPY, LOSS_BINARY_CROSSENTROPY])
        configuration[SCHEDULER] = "ReduceLROnPlateau"
        configuration[SCHEDULER_CONFIGURATION] = dict(patience=8, factor=0.8)
    elif exp == 801:
        # Experiment with big batch size, tempered loss + BCE.
        lr = 3e-4
        batch_size = 256
        model, configuration = generic_experiment(
            configuration,
            llm=DISTILBERT,
            graph=BIG_GCN,
            n=400,
            b=batch_size,
            lr=lr,
            wd=1e-1,
            lora=True,
            quantization=None,
            mixed_precision=True,
            temperature=True
        )
        configuration[LOSS] = ",".join([LOSS_TEMPERED_CROSSENTROPY, LOSS_BINARY_CROSSENTROPY])
        configuration[SCHEDULER] = "ReduceLROnPlateau"
        configuration[SCHEDULER_CONFIGURATION] = dict(patience=8, factor=0.8)


        # configuration[SCHEDULER] = "LambdaLR"
        # lr_lambda = partial(
        #     custom_lr, warmup=40,
        #     lr_init=lr, lr_min=1e-5, lr_tmp=1e-4,
        #     period_oscillation=20,
        #     periods_dampen=5
        # )
        # configuration[SCHEDULER_CONFIGURATION] = dict(lr_lambda=lr_lambda)

    print(configuration)
    return model, configuration
