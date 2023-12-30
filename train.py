from properties import ROOT_DIR, VALIDATION, TEST
import logging
import wandb

from training_loop import training
from evaluation import evaluate_experience
from utils import prepare_experience, get_default_parser, wandb_login
from pathlib import Path


def train_experience(
        exp: int, root_dir: Path = ROOT_DIR,
        debug: bool = False,
        device=None,
        backup_root: Path = None,
        wandb_flag: bool = True
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
    if wandb_flag:
        run = wandb.init(
            project="molecule-nlp",
            entity="molecule-nlp-altegrad-23",
            name=output_directory.name,
            config=configuration,
            reinit=True
        )
    else:
        logging.warning("Weights and biases disabled: logging into the blue!")
    output_directory.mkdir(exist_ok=True, parents=True)
    training(
        model, output_directory, configuration, tokenizer,
        device, print_freq=print_freq,
        backup_folder=backup_folder,
        wandb_flag=wandb_flag
    )
    if wandb_flag:
        run.finish()


if __name__ == '__main__':
    parser = get_default_parser()
    parser.add_argument("-b", "--backup-root", type=Path, default=None, help="Backup root folder")
    parser.add_argument("-w", "--wandb-api-key", type=str, default=None, help="Wandb API key")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--no-eval", action="store_true", help="Disable final evaluation")
    args = parser.parse_args()
    if not args.no_wandb and args.wandb_api_key is not None:
        wandb_login(args.wandb_api_key)
    for exp in args.exp_list:
        train_experience(
            exp,
            debug=args.debug,
            backup_root=args.backup_root,
            device=args.device,
            wandb_flag=not args.no_wandb
        )
        if not args.no_eval:
            for phase in [VALIDATION, TEST]:
                evaluate_experience(exp, backup_root=args.backup_root, device=args.device, phase=phase)
