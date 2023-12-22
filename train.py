from properties import ROOT_DIR
import logging

from training_loop import training
from evaluation import evaluation
from utils import get_device
from experiments import get_experience
from utils import get_default_parser, get_output_directory, get_tokenizer
from pathlib import Path


def prepare_experience(exp: int, root_dir=ROOT_DIR) -> dict:
    model, configuration = get_experience(exp)
    output_directory = get_output_directory(configuration, root_dir=root_dir)
    tokenizer = get_tokenizer(configuration)
    device = get_device()
    return model, configuration, output_directory, tokenizer, device


def train_experience(exp: int, root_dir: Path = ROOT_DIR, debug=False, backup_root: Path = None) -> None:
    if debug:
        print_freq = 1
    else:
        print_freq = 50
    model, configuration, output_directory, tokenizer, device = prepare_experience(exp, root_dir=root_dir)
    if output_directory.exists():
        logging.warning(f"Experience {exp} already trained")
        return
    output_directory.mkdir(exist_ok=True, parents=True)
    backup_folder = None
    if backup_root is not None:
        backup_folder = backup_root/output_directory.name
        backup_folder.mkdir(exist_ok=True, parents=True)
    training(
        model, output_directory, configuration, tokenizer,
        device, print_freq=print_freq,
        backup_folder=backup_folder
    )


def evaluate_experience(exp: int, root_dir=ROOT_DIR) -> None:
    model, configuration, output_directory, tokenizer, device = prepare_experience(exp, root_dir=root_dir)
    evaluation(model, output_directory, configuration, tokenizer, device)


if __name__ == '__main__':
    parser = get_default_parser()
    parser.add_argument("-b", "--backup-root", type=Path, default=None, help="Backup root folder")
    args = parser.parse_args()
    for exp in args.exp_list:
        train_experience(exp, debug=args.debug, backup_root=args.backup_root)
        evaluate_experience(exp)
