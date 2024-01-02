import torch
import argparse
import wandb
from properties import ROOT_DIR, OUT_DIR, ID, NAME, TOKENIZER_NAME, OPTIMIZER, OPTIMIZER_STATE_DICT
from pathlib import Path
from transformers import AutoTokenizer
from typing import Optional, Union
import logging


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def get_output_directory(configuration: dict, root_dir: Path = ROOT_DIR, out_dir: Path = OUT_DIR):
    output_directory = out_dir/f"{configuration[ID]:04d}_{configuration[NAME]}"
    return output_directory


def get_output_directory_experiment(exp: int, out_dir: Path = OUT_DIR) -> Path:
    return sorted(list(out_dir.glob(f"{exp:04d}_*")))[-1]


def get_tokenizer(configuration):
    tokenizer_model_name = configuration[TOKENIZER_NAME]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
    return tokenizer


def get_default_parser(help="Train models") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("-d", "--device", type=str,
                        choices=["cpu", "cuda"], default=str(get_device()), help="Training device")
    parser.add_argument("-e", "--exp-list", nargs="+", type=int, default=[1], help="List of experiments to run")
    parser.add_argument("-dbg", "--debug", action="store_true", help="Debug mode")

    return parser


def wandb_login(wandb_api_key: Optional[str] = None):
    wandb.login(key=wandb_api_key)


def reload_model_and_optimizer_state(
    experiment: Union[int, Path],
    out_dir: Path = OUT_DIR,
    backup_root: Path = None,
    checkpoint: Optional[Union[int, str]] = None,
    configuration: dict = {},
    model: torch.nn.Module = None
):
    if experiment is not None:
        if isinstance(experiment, Path):
            output_directory = experiment
        else:
            dir_to_check = [out_dir]
            if backup_root is not None:
                dir_to_check.append(backup_root)
            for current_out_dir in dir_to_check:
                if isinstance(experiment, int):
                    output_directory = get_output_directory_experiment(experiment, out_dir=current_out_dir)
                elif isinstance(experiment, str):
                    output_directory = out_dir/experiment
                if output_directory.exists():
                    break
    assert output_directory.exists(), f"Output directory {out_dir} does not exist"
    if checkpoint is None:
        pretrained_model_path_list = sorted(list(output_directory.glob("*.pt")))
        assert len(pretrained_model_path_list) > 0, f"No model checkpoint found at {output_directory}"
        pretrained_model_path = pretrained_model_path_list[-1]

    else:
        if isinstance(checkpoint, int):
            pretrained_model_path = output_directory/f"model_{checkpoint:04d}.pt"
        elif isinstance(checkpoint, str):
            pretrained_model_path = output_directory/checkpoint
        else:
            raise ValueError(f"Invalid checkpoint type {checkpoint}")
    assert pretrained_model_path.exists(), f"Checkpoint {pretrained_model_path} does not exist"
    logging.warning(f"Reloading model from {pretrained_model_path}")
    check_point = torch.load(pretrained_model_path, map_location='cpu')
    model.load_state_dict(check_point['model_state_dict'])
    configuration[OPTIMIZER][OPTIMIZER_STATE_DICT] = check_point['optimizer_state_dict']
    return model, configuration
