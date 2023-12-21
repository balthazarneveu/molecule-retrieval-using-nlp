from properties import (
    NB_EPOCHS, BATCH_SIZE, LEARNING_RATE, TOKENIZER_NAME, NAME, ANNOTATIONS, WEIGHT_DECAY, BETAS, OPTIMIZER,
    TRAIN, VALIDATION, TEST, ID, MAX_STEP_PER_EPOCH
)
import torch
from model import Model
from typing import Tuple


def get_experience(exp: int) -> Tuple[torch.nn.Module, dict]:
    configuration = {
        NB_EPOCHS: 5,
        # batch_size : 32
        BATCH_SIZE: (16, 8, 8),  # To fit Nvidia T500 4Gb RAM
        OPTIMIZER: {
            LEARNING_RATE: 2e-5,
            WEIGHT_DECAY: 0.01,
            BETAS: (0.9, 0.999)
        },
        TOKENIZER_NAME: 'distilbert-base-uncased',
        MAX_STEP_PER_EPOCH: None
    }
    if exp == 0:
        configuration[NAME] = 'check-pipeline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Check pipeline'
        configuration[MAX_STEP_PER_EPOCH] = 5
        model = Model(model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
                      nhid=8, graph_hidden_channels=8)
    if exp == 1:
        configuration[NAME] = 'Baseline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Baseline - provided by organizers'
        model = Model(model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
                      nhid=300, graph_hidden_channels=300)  # nout = bert model hidden dim
    if exp == 2:
        configuration[NB_EPOCHS] = 30
        configuration[NAME] = 'Baseline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Baseline - provided by organizers - 30 epochs'
        model = Model(model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
                      nhid=300, graph_hidden_channels=300)  # nout = bert model hidden dim
        configuration[MAX_STEP_PER_EPOCH] = 5
    configuration[ID] = exp
    configuration[BATCH_SIZE] = {
        TRAIN: configuration[BATCH_SIZE][0],
        VALIDATION: configuration[BATCH_SIZE][1],
        TEST: configuration[BATCH_SIZE][2]
    }
    return model, configuration
