from properties import (
    NB_EPOCHS, BATCH_SIZE, LEARNING_RATE, TOKENIZER_NAME, WEIGHT_DECAY, BETAS, OPTIMIZER,
    TRAIN, VALIDATION, TEST, ID, MAX_STEP_PER_EPOCH, PLATFORM, MODEL_SIZE, SHA1, OPTIMIZER_STATE_DICT,
    ROOT_DIR
)
import torch
from experiments_round_00 import get_baseline_experience
from experiments_round_01 import get_round_1_experience
from experiments_round_02 import get_round_2_experience
from experiments_round_04 import get_round_4_experience
from experiments_round_80 import get_round_80_experience
from experiments_round_90 import get_round_90_experience
from typing import Tuple
from platform_description import get_hardware_descriptor, get_git_sha1
from pathlib import Path
from utils import get_device, get_tokenizer, get_output_directory


def get_experience(exp: int, root_dir: Path = None, backup_root: Path = None) -> Tuple[torch.nn.Module, dict]:
    """
    Full definition of experiments roadmap
    https://github.com/balthazarneveu/molecule-retrieval-using-nlp/issues
    """
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
    elif exp >= 100 and exp <= 199:
        get_exp = get_round_1_experience
    elif exp >= 200 and exp <= 299:
        get_exp = get_round_2_experience
    elif exp >= 400 and exp <= 499:
        get_exp = get_round_4_experience
    elif exp >= 8000 and exp <= 8999:
        get_exp = get_round_80_experience
    elif exp >= 9000 and exp <= 9999:
        get_exp = get_round_90_experience
    model, configuration = get_exp(exp, configuration, root_dir=root_dir, backup_root=backup_root)
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
    optimizer_state_dict = configuration[OPTIMIZER].get(OPTIMIZER_STATE_DICT, None)
    if optimizer_state_dict is not None:
        configuration["optimizer_initial_state_dict"] = "reloaded"
    else:
        configuration["optimizer_initial_state_dict"] = "empty"
    if OPTIMIZER_STATE_DICT in configuration[OPTIMIZER]:
        configuration[OPTIMIZER].pop(OPTIMIZER_STATE_DICT)  # to avoid writing it in the configuation file
    return model, configuration, optimizer_state_dict


def prepare_experience(
        exp: int, root_dir: Path = ROOT_DIR, device=None, backup_root: Path = None
) -> Tuple[torch.nn.Module, dict, Path, any, torch.device, Path]:
    model, configuration, optimizer_state_dict = get_experience(exp, root_dir=root_dir, backup_root=backup_root)
    output_directory = get_output_directory(configuration, root_dir=root_dir)
    tokenizer = get_tokenizer(configuration)
    if device is None:
        device = get_device()
        print(f"Device not specified, using default one {device}")
    backup_folder = None
    if backup_root is not None:
        backup_folder = backup_root/output_directory.name
        backup_folder.mkdir(exist_ok=True, parents=True)
    return model, configuration, output_directory, tokenizer, device, backup_folder, optimizer_state_dict


if __name__ == "__main__":
    from utils import get_default_parser
    import yaml
    args = get_default_parser(help="Describe experiments").parse_args()
    for exp_id in args.exp_list:
        model, config = get_experience(exp_id)
        print(yaml.dump(config, default_flow_style=False))
