from properties import (
    NB_EPOCHS, BATCH_SIZE, LEARNING_RATE, TOKENIZER_NAME, NAME, ANNOTATIONS, WEIGHT_DECAY, OPTIMIZER,
    SCHEDULER, SCHEDULER_CONFIGURATION,
    PLATEAU, COSINE_WARMUP,
    SCIBERT, DISTILBERT,
    BASE_GCN, BIG_GCN, FAT_GCN
)
from multimodal_model import MultimodalModel
from language_model import TextEncoder
from graph_model import BasicGraphEncoder, BigGraphEncoder, FatGraphEncoder
from lora import get_lora_configuration, get_quantization_configuration
import torch
import logging
from typing import Tuple, Optional, Union
import numpy as np
LLM_SHORT_NAMES = [DISTILBERT, SCIBERT]
GNN_SHORT_NAMES = [BASE_GCN, BIG_GCN, FAT_GCN]

SCHEDULER_SHORT_NAMES = [PLATEAU, COSINE_WARMUP]


def generic_experiment(
    configuration: dict,
    b: int = 32,
    n: int = 150,
    lr: float = 7e-6,
    wd: float = 0.,
    llm: Optional[Union[str, torch.nn.Module]] = DISTILBERT,
    lora: Optional[bool] = False,
    quantization: Optional[str] = None,
    graph: Optional[Union[str, torch.nn.Module]] = BASE_GCN,
    scheduler: Optional[str] = None,
    scheduler_configuration: Optional[dict] = None,
) -> Tuple[torch.nn.Module, dict]:
    # ------------------------------------------------------------------------------------ HYPERPARAMETERS
    configuration[NB_EPOCHS] = n
    configuration[OPTIMIZER][LEARNING_RATE] = lr
    configuration[OPTIMIZER][WEIGHT_DECAY] = wd
    configuration[BATCH_SIZE] = (b, b, b)

    # ------------------------------------------------------------------------------------ SCHEDULER
    if scheduler is not None and isinstance(scheduler, str):
        assert scheduler in SCHEDULER_SHORT_NAMES, f"{scheduler} must be in {SCHEDULER_SHORT_NAMES}"
        # Template schedulers
        if scheduler == PLATEAU:
            configuration[SCHEDULER] = "ReduceLROnPlateau"
            configuration[SCHEDULER_CONFIGURATION] = dict(patience=5, factor=0.5)
        elif scheduler == COSINE_WARMUP:
            configuration[SCHEDULER] = "CosineAnnealingWarmRestarts"
            configuration[SCHEDULER_CONFIGURATION] = dict(T_0=30, T_mult=1, eta_min=1e-5)
        if scheduler_configuration is not None:
            configuration[SCHEDULER_CONFIGURATION] = scheduler_configuration
    # ------------------------------------------------------------------------------------ LLM  DEFINITION
    if isinstance(llm, str):  # PREDEFINED LLM
        assert llm in LLM_SHORT_NAMES, f"{llm} must be in {LLM_SHORT_NAMES}"
        configuration[TOKENIZER_NAME] = llm
        if llm == DISTILBERT:
            configuration[TOKENIZER_NAME] = "distilbert-base-uncased"
            configuration[NAME] = 'BERT'
            configuration[ANNOTATIONS] = 'Distil BERT'
        elif llm == SCIBERT:
            configuration[TOKENIZER_NAME] = "allenai/scibert_scivocab_uncased"
            configuration[NAME] = 'SciBERT-GCN'
            configuration[ANNOTATIONS] = 'SciBERT'
        # LORA & Quantitzation
        q_dict = None
        if lora:
            lora_dict = get_lora_configuration(configuration[TOKENIZER_NAME])
            configuration[NAME] = 'Lora'+configuration[NAME]
            configuration[ANNOTATIONS] = 'Lora ' + configuration[ANNOTATIONS]
            configuration["finetuning_mode"] = "LoRA"
        else:
            lora_dict = None
        if quantization is not None:
            assert lora, "Quantization only works with LORA so far"
            if quantization == "nf4":
                q_dict = get_quantization_configuration()
                configuration[NAME] = configuration[NAME].replace("Lora", "QLora")
                configuration[ANNOTATIONS] = configuration[ANNOTATIONS].replace("Lora", "QLora")
                configuration["quantization"] = quantization
            else:
                raise NameError(f"Quantization {quantization} not implemented")
        text_encoder = TextEncoder(configuration[TOKENIZER_NAME], freeze=False,
                                   lora=lora_dict, quantization_config=q_dict)
    else:  # CUSTOM LLM INSTANCE
        text_encoder = llm
        logging.warning(f"Self defined Large Language Model {llm}!!!!")
    # ------------------------------------------------------------------------------------ GRAPH DEFINITION
    if isinstance(graph, str):
        assert graph in GNN_SHORT_NAMES, f"{graph} must be in {GNN_SHORT_NAMES}"
        gcn_out = 768
        gcn_in = 300
        if graph == BASE_GCN:
            depth = 3
            gr_ghid = 300
            fc_hid = 300
            graph_encoder = BasicGraphEncoder(num_node_features=gcn_in, nout=gcn_out,
                                              nhid=fc_hid, graph_hidden_channels=gr_ghid)
            configuration[NAME] += "-GCN"
            configuration[ANNOTATIONS] += "- base GCN"
        elif graph == BIG_GCN:
            depth = 5
            gr_ghid = 512
            fc_hid = 256
            configuration[ANNOTATIONS] += "- bigger GCN"
            configuration[NAME] += "-biggerGCN"
            graph_encoder = BigGraphEncoder(num_node_features=gcn_in, nout=gcn_out,
                                            nhid=fc_hid, graph_hidden_channels=gr_ghid)
        elif graph == FAT_GCN:
            depth = 7
            gr_ghid = 512
            fc_hid = 256
            configuration[ANNOTATIONS] += "-fatGCN"
            configuration[NAME] += " - FatGCN"
            graph_encoder = FatGraphEncoder(num_node_features=gcn_in, nout=gcn_out,
                                            nhid=fc_hid, graph_hidden_channels=gr_ghid)
        configuration["GCN-architecture"] = {
            "depth": depth,
            "GCN-FC-hidden-size": gr_ghid,
            "GCN-hidden-size": fc_hid,
            "GNN-out-size": gcn_out,
        }
    else:
        logging.warning(f"Self defined Large Language Model {llm}!!!!")
        graph_encoder = graph
    # ------------------------------------------------------------------------------------ MODEL DEFINITION
    model = MultimodalModel(graph_encoder, text_encoder)

    return model, configuration


def custom_lr(
    epoch,
    warmup: int = 50, lr_init: float = 5e-4, lr_min: float = 1e-5,
    lr_tmp: float = 1e-4,
    period_oscillation: int = 20, periods_dampen: int = 5
):
    """To be used with partial
    """
    if epoch < warmup:
        oscillator = (np.cos(epoch/warmup * np.pi) + 1)/2
        return (lr_min + oscillator*(lr_init-lr_min))/lr_init
    else:
        t = (epoch-warmup)

        t_mod = t % period_oscillation

        iter = t//period_oscillation
        oscillator = (np.cos(t_mod/(period_oscillation) * np.pi) + 1)/2
        factor_modulation = max(0, ((periods_dampen-iter)/periods_dampen))
        amplitude = factor_modulation*(lr_tmp-lr_min)
        return (lr_min + oscillator*amplitude)/lr_init
