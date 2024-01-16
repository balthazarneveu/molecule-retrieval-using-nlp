from properties import (
    NB_EPOCHS, BATCH_SIZE, LEARNING_RATE, TOKENIZER_NAME, NAME, ANNOTATIONS, WEIGHT_DECAY, OPTIMIZER,
)
from pathlib import Path
from multimodal_model import MultimodalModel
from language_model import TextEncoder
from graph_model import BasicGraphEncoder
from peft import TaskType
import torch
from typing import Tuple


def get_load_configuration(model_name: str) -> dict:
    if "distil" in model_name:
        target_modules = {
            "q_lin",
            "k_lin",
            "v_lin",
            "out_lin",
        }
    else:
        target_modules = {
            "query",
            "key",
            "value",
            "dense",
        }
    lora_dict = dict(
        task_type=TaskType.CAUSAL_LM,
        # task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        target_modules=target_modules,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    return lora_dict


SHORT_NAMES = ["distilbert", "scibert", "bert"]


def lora_exp(
    configuration: dict,
    b: int = 32,
    n: int = 150,
    lr: float = 7e-6,
    wd: float = 0.1,
    model_name: str = "distilbert"
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
        configuration[ANNOTATIONS] = 'Trainable Lora Distil BERT - base GCN'
    if model_name == "scibert":
        configuration[TOKENIZER_NAME] = "allenai/scibert_scivocab_uncased"
        configuration[NAME] = 'LoraSciBERT-GCN'
        configuration[ANNOTATIONS] = 'Lora SciBERT - base GCN'
    lora_dict = get_load_configuration(configuration[TOKENIZER_NAME])
    graph_encoder = BasicGraphEncoder(num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300)
    text_encoder = TextEncoder(configuration[TOKENIZER_NAME], freeze=False, lora=lora_dict)
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
    assert exp >= 500 and exp <= 599, "round 5 between 500 and 599"
    if exp == 500:
        model, conf = lora_exp(conf, b=32, n=150, lr=7e-6, wd=0.1, model_name="distilbert")
    elif exp == 501:
        model, conf = lora_exp(conf, b=32, n=60, lr=7e-6, wd=0.1, model_name="scibert")
    elif exp == 502:
        model, conf = lora_exp(conf, b=256, n=200, lr=7e-6, wd=0.1, model_name="scibert") # probably A5000 required
    return model, conf
