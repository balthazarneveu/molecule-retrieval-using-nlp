import torch
import argparse
from properties import ROOT_DIR, OUT_DIR, ID, NAME, TOKENIZER_NAME
from pathlib import Path
from transformers import AutoTokenizer
from pathlib import Path


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


def parse_args(help="Train models"):
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("-d", "--device", type=str,
                        choices=["cpu", "cuda"], default=str(get_device()), help="Training device")
    parser.add_argument("-e", "--exp-list", nargs="+", type=int, default=[1], help="List of experiments to run")
    parser.add_argument("-dbg", "--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    return args
