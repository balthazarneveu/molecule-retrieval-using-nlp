from properties import ROOT_DIR
import logging

from training_loop import training
from evaluation import evaluation
from utils import get_device
from experiments import get_experience
from utils import get_default_parser, get_output_directory, get_tokenizer


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


# def plot_metrics_experience(exp: int, root_dir=ROOT_DIR) -> None:
#     _model, configuration, output_directory, _tokenizer, _device = prepare_experience(exp, root_dir=root_dir)
#     plot_metrics_experience(output_directory, configuration)


if __name__ == '__main__':
    args = get_default_parser().parse_args
    for exp in args.exp_list:
        train_experience(exp, debug=args.debug)
        evaluate_experience(exp)
