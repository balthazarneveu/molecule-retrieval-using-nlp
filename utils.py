import torch
import argparse
import wandb
from properties import ROOT_DIR, OUT_DIR, ID, NAME, TOKENIZER_NAME
from pathlib import Path
from transformers import AutoTokenizer
from experiments import get_experience
from typing import Tuple, Optional


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


def prepare_experience(
        exp: int, root_dir: Path = ROOT_DIR, device=None, backup_root: Path = None
) -> Tuple[torch.nn.Module, dict, Path, AutoTokenizer, torch.device, Path]:
    model, configuration = get_experience(exp, root_dir=root_dir, backup_root=backup_root)
    output_directory = get_output_directory(configuration, root_dir=root_dir)
    tokenizer = get_tokenizer(configuration)
    if device is None:
        device = get_device()
        print(f"Device not specified, using default one {device}")
    backup_folder = None
    if backup_root is not None:
        backup_folder = backup_root/output_directory.name
        backup_folder.mkdir(exist_ok=True, parents=True)
    return model, configuration, output_directory, tokenizer, device, backup_folder
