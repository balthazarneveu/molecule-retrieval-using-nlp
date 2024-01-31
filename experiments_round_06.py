from properties import (
    SCHEDULER, SCHEDULER_CONFIGURATION,
    DISTILBERT, SCIBERT, BIG_GCN, FAT_GCN, PLATEAU,
    LOSS, NAME
)
from pathlib import Path
from properties import OUT_DIR
from experiments_generic import generic_experiment, custom_lr
from functools import partial
import torch


def get_round_6_experience(exp: int, configuration: dict, root_dir: Path = None, backup_root: Path = None):
    """"""
    assert exp >= 600 and exp < 699, f"Experiment {exp} is not in the round 90"
    # configuration["max_step_per_epoch"] = 5
    if exp == 600:
        lr = 4e-4
        model, configuration = generic_experiment(
            configuration,
            llm=DISTILBERT, graph=BIG_GCN,
            n=200,
            b=32, lr=lr, wd=1e-1,
            lora=True, quantization=None
        )
        configuration[SCHEDULER] = "LambdaLR"
        lr_lambda = partial(
            custom_lr, warmup=40,
            lr_init=lr, lr_min=1e-5, lr_tmp=1e-4,
            period_oscillation=20,
            periods_dampen=5
        )
        configuration[SCHEDULER_CONFIGURATION] = dict(lr_lambda=lr_lambda)
    elif exp == 601:
        # configuration["max_step_per_epoch"] = 5
        lr = 4e-4
        model, configuration = generic_experiment(
            configuration,
            llm=DISTILBERT, graph=BIG_GCN,
            n=200,
            b=32, lr=lr, wd=1e-1,
            lora=True, quantization=None,
            temperature=True,
        )
        configuration[SCHEDULER] = "ReduceLROnPlateau"
        configuration[SCHEDULER_CONFIGURATION] = dict(patience=8, factor=0.8)
        configuration[LOSS] = "Tempered"
    elif exp == 603:
        # 573
        lr = 4e-4
        model, configuration = generic_experiment(
            configuration,
            llm=SCIBERT, graph=FAT_GCN,
            n=200,
            b=256, lr=lr, wd=1e-1,
            lora=True, quantization=None,
            temperature=False,
        )
        reload_model = OUT_DIR/"0573_LoraSciBERT-FatGCN"/"model_0145.pt"
        model.load_state_dict(torch.load(reload_model)["model_state_dict"])
        for param in model.text_encoder.bert.parameters():
            param.requires_grad = False
        configuration[SCHEDULER] = "ReduceLROnPlateau"
        configuration[SCHEDULER_CONFIGURATION] = dict(patience=8, factor=0.8)
        configuration[NAME] += " Pretrained on 573"
    print(configuration)
    return model, configuration
