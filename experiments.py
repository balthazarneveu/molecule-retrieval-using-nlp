from properties import (
    NB_EPOCHS, BATCH_SIZE, LEARNING_RATE, TOKENIZER_NAME, WEIGHT_DECAY, BETAS, OPTIMIZER,
    TRAIN, VALIDATION, TEST, ID, MAX_STEP_PER_EPOCH, PLATFORM, MODEL_SIZE, SHA1
)
import torch
from experiments_round_00 import get_baseline_experience
from experiments_round_01 import get_round_1_experience
from experiments_round_80 import get_round_80_experience
from experiments_round_90 import get_round_90_experience
from typing import Tuple
from platform_description import get_hardware_descriptor, get_git_sha1
from pathlib import Path


def get_experience(exp: int, root_dir: Path = None, backup_root: Path = None) -> Tuple[torch.nn.Module, dict]:
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
    # Manual configurations
    if exp >= 0 and exp <= 99:
        get_exp = get_baseline_experience
    elif exp >= 100 and exp <= 200:
        get_exp = get_round_1_experience
    elif exp >= 8000 and exp <= 8999:
        get_exp = get_round_80_experience
    elif exp >= 9000 and exp <= 9999:
        get_exp = get_round_90_experience
    model, configuration = get_exp(exp, configuration, root_dir, backup_root)
    # Auto assign configuration values
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
