from properties import (
    NB_EPOCHS, BATCH_SIZE, LEARNING_RATE, TOKENIZER_NAME, NAME, ANNOTATIONS, WEIGHT_DECAY, BETAS, OPTIMIZER,
    TRAIN, VALIDATION, TEST, ID, MAX_STEP_PER_EPOCH, PLATFORM, MODEL_SIZE, SHA1
)
import torch
from model import Model
from typing import Tuple
from platform_description import get_hardware_descriptor, get_git_sha1


def get_experience(exp: int) -> Tuple[torch.nn.Module, dict]:
    configuration = {
        NB_EPOCHS: 5,
        BATCH_SIZE: [16, 8, 8],  # To fit Nvidia T500 4Gb RAM
        OPTIMIZER: {
            LEARNING_RATE: 2e-5,
            WEIGHT_DECAY: 0.01,
            BETAS: [0.9, 0.999]
        },
        TOKENIZER_NAME: 'distilbert-base-uncased',
        MAX_STEP_PER_EPOCH: None
    }
    if exp == 0:
        configuration[NAME] = 'check-pipeline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Check pipeline'
        configuration[MAX_STEP_PER_EPOCH] = 60
        model = Model(model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
                      nhid=8, graph_hidden_channels=8)
    if exp == 1:
        configuration[NB_EPOCHS] = 5
        configuration[NAME] = 'Baseline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Baseline - provided by organizers'
        model = Model(model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
                      nhid=300, graph_hidden_channels=300)  # nout = bert model hidden dim
    if exp == 2:
        configuration[NB_EPOCHS] = 5
        configuration[NAME] = 'Baseline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Baseline - provided by organizers - 5 epochs'
        model = Model(model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
                      nhid=300, graph_hidden_channels=300)  # nout = bert model hidden dim
    if exp == 3:
        configuration[NB_EPOCHS] = 30
        configuration[NAME] = 'Baseline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Baseline - provided by organizers - 30 epochs'
        model = Model(model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
                      nhid=300, graph_hidden_channels=300)  # nout = bert model hidden dim
    if exp == 4:
        configuration[BATCH_SIZE] = (32, 64, 64)  # Collab
        configuration[NB_EPOCHS] = 30
        configuration[NAME] = 'Baseline-BERT-GCN'
        configuration[ANNOTATIONS] = 'Baseline - provided by organizers - Collab'
        model = Model(model_name=configuration[TOKENIZER_NAME], num_node_features=300, nout=768,
                      nhid=300, graph_hidden_channels=300)  # nout = bert model hidden dim
    configuration[ID] = exp
    configuration[PLATFORM] = get_hardware_descriptor()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    configuration[MODEL_SIZE] = total_params
    configuration[BATCH_SIZE] = {
        TRAIN: configuration[BATCH_SIZE][0],
        VALIDATION: configuration[BATCH_SIZE][1],
        TEST: configuration[BATCH_SIZE][2]
    }
    configuration[SHA1] = get_git_sha1()
    return model, configuration


if __name__ == "__main__":
    from utils import get_default_parser
    import yaml
    args = get_default_parser(help="Describe experiments").parse_args()
    for exp_id in args.exp_list:
        model, config = get_experience(exp_id)
        print(yaml.dump(config, default_flow_style=False))
