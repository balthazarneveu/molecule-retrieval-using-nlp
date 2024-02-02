from properties import (
    SCHEDULER, SCHEDULER_CONFIGURATION,
    DISTILBERT, SCIBERT, BIG_GCN, FAT_GCN, PLATEAU,
    LOSS, NAME, BATCH_SIZE, LOSS_TEMPERED_CROSSENTROPY, LOSS_BINARY_CROSSENTROPY
)
from pathlib import Path
from properties import OUT_DIR
from experiments_generic import generic_experiment, custom_lr
from graph_model import FatGraphEncoder
from functools import partial
import torch
from huggingface_hub import hf_hub_download


def get_round_6_experience(exp: int, configuration: dict, root_dir: Path = None, backup_root: Path = None):
    """"""
    assert exp >= 600 and exp < 699, f"Experiment {exp} is not in the round 6"
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
    elif exp == 601 or exp == 620 or exp == 621:
        # configuration["max_step_per_epoch"] = 5
        lr = 4e-4
        model, configuration = generic_experiment(
            configuration,
            llm=DISTILBERT, graph=BIG_GCN,
            n=200 if exp == 601 else 65,
            b=32 if exp == 601 else 64,
            lr=lr, wd=1e-1,
            lora=True, quantization=None,
            temperature=True,
        )
        if exp == 621:
            REPO_ID = "balthou/0620_LoraBERT-biggerGCN"
            FILENAME = "model_0058.pt"
            reload_model = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, cache_dir=OUT_DIR)
            model.load_state_dict(torch.load(reload_model)["model_state_dict"])
        configuration[SCHEDULER] = "ReduceLROnPlateau"
        configuration[SCHEDULER_CONFIGURATION] = dict(patience=8, factor=0.8)
        configuration[LOSS] = LOSS_TEMPERED_CROSSENTROPY
    elif exp == 603 or exp == 604:
        # 573
        lr = 4e-4
        if exp == 603:
            batch_size = 256
            n = 200
        elif exp == 604:
            batch_size = 600
            n = 65
        model, configuration = generic_experiment(
            configuration,
            llm=SCIBERT, graph=FAT_GCN,
            n=n,
            b=batch_size, lr=lr, wd=1e-1,
            lora=True, quantization=None,
            temperature=False,
        )
        REPO_ID = "balthou/molnlp_0573_LoraSciBERT_FatGCN"
        FILENAME = "model_0145.pt"
        reload_model = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, cache_dir=OUT_DIR)
        # reload_model = OUT_DIR/"0573_LoraSciBERT-FatGCN"/"model_0145.pt"
        model.load_state_dict(torch.load(reload_model)["model_state_dict"])
        for param in model.text_encoder.bert.parameters():
            param.requires_grad = False
        configuration[SCHEDULER] = "ReduceLROnPlateau"
        configuration[SCHEDULER_CONFIGURATION] = dict(patience=8, factor=0.8)
        configuration[NAME] += " Pretrained on 573"
    if exp == 610 or exp == 611:
        # 573 LLM
        lr = 1e-3
        batch_size = 128
        n = 200
        model, configuration = generic_experiment(
            configuration,
            llm=SCIBERT, graph=FAT_GCN,
            n=n,
            b=batch_size, lr=lr, wd=1e-1,
            lora=True, quantization=None,
            temperature=False,
        )
        REPO_ID = "balthou/molnlp_0573_LoraSciBERT_FatGCN"
        FILENAME = "model_0145.pt"
        reload_model = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, cache_dir=OUT_DIR)
        model.load_state_dict(torch.load(reload_model)["model_state_dict"])
        for param in model.text_encoder.bert.parameters():
            param.requires_grad = False
        graph_encoder = FatGraphEncoder(num_node_features=300, nout=768, nhid=512, graph_hidden_channels=512)
        model.graph_encoder = graph_encoder
        if exp == 611:
            configuration[LOSS] = LOSS_BINARY_CROSSENTROPY
        configuration[SCHEDULER] = "ReduceLROnPlateau"
        configuration[SCHEDULER_CONFIGURATION] = dict(patience=5, factor=0.5)
        configuration[NAME] += " Pretrained on 573"

    elif exp == 630:
        # 9009 Lea competition of loss
        model, configuration = generic_experiment(
            configuration,
            llm=DISTILBERT,
            graph=FAT_GCN,
            n=200,
            b=128,
            lr=5e-4,
            wd=1e-1,
            scheduler=PLATEAU,
            scheduler_configuration=dict(patience=8, factor=0.5),
            lora=False,
            quantization=None,
            temperature=True
        )
        configuration[BATCH_SIZE] = (128, 32, 32)
        configuration[SCHEDULER] = "ReduceLROnPlateau"
        configuration[SCHEDULER_CONFIGURATION] = dict(patience=8, factor=0.8)
        configuration[LOSS] = LOSS_TEMPERED_CROSSENTROPY
    print(configuration)
    return model, configuration
