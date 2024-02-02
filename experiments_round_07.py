from properties import (
    SCHEDULER, SCHEDULER_CONFIGURATION,
    DISTILBERT, SCIBERT, BIG_GCN, FAT_GCN, PLATEAU,
    LOSS, NAME, BATCH_SIZE, MAX_STEP_PER_EPOCH, LOSS_TEMPERED_CROSSENTROPY
)
from pathlib import Path
from properties import OUT_DIR
from experiments_generic import generic_experiment, custom_lr
from graph_model import FatGraphEncoder
from functools import partial
import torch
from huggingface_hub import hf_hub_download


def get_round_7_experience(exp: int, configuration: dict, root_dir: Path = None, backup_root: Path = None):
    """"""
    assert exp >= 700 and exp < 799, f"Experiment {exp} is not in the round 7"
    if exp == 700:
        # 9011 baseline
        lr = 3e-4
        batch_size = 180
        n = 200
        model, configuration = generic_experiment(
            configuration,
            llm=DISTILBERT, graph=FAT_GCN,
            n=n,
            b=batch_size,
            lr=lr,
            wd=1e-1,
            scheduler=PLATEAU, scheduler_configuration=dict(patience=8, factor=0.5),
            lora=False, quantization=None
        )
        # REPO_ID = "balthou/9011_BERT_biggerGCN"
        # FILENAME = "model_0202.pt"

        REPO_ID = "balthou/9009_BERT_FatGCN"
        FILENAME = "model_0190.pt"

        reload_model = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, cache_dir=OUT_DIR)
        model.load_state_dict(torch.load(reload_model)["model_state_dict"])
        for param in model.text_encoder.bert.parameters():
            param.requires_grad = False
        configuration[SCHEDULER] = "ReduceLROnPlateau"
        configuration[SCHEDULER_CONFIGURATION] = dict(patience=8, factor=0.8)
        # configuration[NAME] += " Pretrained on 9011"
        configuration[NAME] += " Pretrained on 9009"
    elif exp == 701:
        # configuration[MAX_STEP_PER_EPOCH] = 4
        batch_size = 64
        n = 300
        lr = 1e-3
        model, configuration = generic_experiment(
            configuration,
            llm=DISTILBERT,
            graph=BIG_GCN,
            n=n,
            b=batch_size, lr=lr, wd=1e-1,
            lora=True,
            # quantization="nf4",
            temperature=True,
        )
        configuration[LOSS] = LOSS_TEMPERED_CROSSENTROPY
        configuration["use_amp"] = True
    print(configuration)
    return model, configuration
