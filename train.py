from properties import (
    TOKENIZER_NAME, NAME, ID, ROOT_DIR
)
import logging
import argparse
from transformers import AutoTokenizer
from pathlib import Path
from training_loop import training
from evaluation import evaluation
from utils import get_device
from experiments import get_experience
from metrics import plot_metrics


def get_tokenizer(configuration):
    tokenizer_model_name = configuration[TOKENIZER_NAME]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
    return tokenizer


def get_output_directory(configuration: dict, root_dir: Path = ROOT_DIR):
    output_directory = root_dir/'__output'/f"{configuration[ID]:04d}_{configuration[NAME]}"
    return output_directory


def prepare_experience(exp: int, root_dir=ROOT_DIR) -> dict:
    model, configuration = get_experience(exp)
    output_directory = get_output_directory(configuration, root_dir=root_dir)
    tokenizer = get_tokenizer(configuration)
    device = get_device()
    return model, configuration, output_directory, tokenizer, device


def train_experience(exp: int, root_dir=ROOT_DIR, debug=False) -> None:
    if debug:
        print_freq = 1
    model, configuration, output_directory, tokenizer, device = prepare_experience(exp, root_dir=root_dir)
    if output_directory.exists():
        logging.warning(f"Experience {exp} already trained")
        return
    output_directory.mkdir(exist_ok=True, parents=True)
    training(model, output_directory, configuration, tokenizer, device, print_freq=print_freq)


def evaluate_experience(exp: int, root_dir=ROOT_DIR) -> None:
    model, configuration, output_directory, tokenizer, device = prepare_experience(exp, root_dir=root_dir)
    evaluation(model, output_directory, configuration, tokenizer, device)


def plot_metrics_experience(exp: int, root_dir=ROOT_DIR) -> None:
    _model, configuration, output_directory, _tokenizer, _device = prepare_experience(exp, root_dir=root_dir)
    plot_metrics_experience(output_directory, configuration)


def parse_args():
    parser = argparse.ArgumentParser(description="Train classification models on Abide dataset - compare performances")
    parser.add_argument("-d", "--device", type=str,
                        choices=["cpu", "cuda"], default=str(get_device()), help="Training device")
    parser.add_argument("-e", "--exp-list", nargs="+", type=int, default=[1], help="List of experiments to run")
    parser.add_argument("-dbg", "--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    for exp in args.exp_list:
        train_experience(exp, debug=args.debug)
        evaluate_experience(exp)
