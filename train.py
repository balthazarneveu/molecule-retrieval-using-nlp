from properties import ROOT_DIR, VALIDATION, TEST
import logging

from training_loop import training
from evaluation import evaluate_experience
from utils import prepare_experience, get_default_parser
from pathlib import Path


def train_experience(
        exp: int, root_dir: Path = ROOT_DIR,
        debug: bool = False,
        device=None,
        backup_root: Path = None
) -> None:
    if debug:
        print_freq = 5
    else:
        print_freq = 50
    model, configuration, output_directory, tokenizer, device, backup_folder = prepare_experience(
        exp,
        root_dir=root_dir, device=device,
        backup_root=backup_root
    )
    if output_directory.exists():
        logging.warning(f"Experience {exp} already trained")
        return
    output_directory.mkdir(exist_ok=True, parents=True)
    training(
        model, output_directory, configuration, tokenizer,
        device, print_freq=print_freq,
        backup_folder=backup_folder
    )


if __name__ == '__main__':
    parser = get_default_parser()
    parser.add_argument("-b", "--backup-root", type=Path, default=None, help="Backup root folder")
    args = parser.parse_args()
    for exp in args.exp_list:
        train_experience(exp, debug=args.debug, backup_root=args.backup_root, device=args.device)
        for phase in [VALIDATION, TEST]:
            evaluate_experience(exp, backup_root=args.backup_root, device=args.device, phase=phase)