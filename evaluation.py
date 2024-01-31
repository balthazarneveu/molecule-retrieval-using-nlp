from properties import (
    BATCH_SIZE,
    TRAIN, TEST, VALIDATION, DATA_DIR, PLATFORM,
    ROOT_DIR,
    TOKENIZER_NAME
)
from torch_geometric.data import DataLoader
from dataloader import GraphDataset, TextDataset
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader as TorchDataLoader
from validation import eval
from dataloader import GraphTextDataset
import pandas as pd
from utils import get_default_parser
from experiments import prepare_experience
from platform_description import get_git_sha1
from tqdm import tqdm
import logging


def evaluation(
    model: torch.nn.Module, model_path: Path, configuration: dict, tokenizer, device: str,
    backup_folder: Path = None, override: bool = False,
    phase: str = TEST,
    forced_batch_size: int = None
):
    """Prepare results on a test set to be submitted to the Kaggle competition

    Args:
        model (torch.nn.Module): torch model to be evaluated
        model_path (Path): folder with trained model weights
        configuration (dict): configuration dictionary
        tokenizer: Required to encode text
        device (str): cpu or cuda
        backup_folder (Path, optional): Backup for Collab. Defaults to None.
    """
    csv_name = {
        TEST: 'submission.csv',
        VALIDATION: 'validation.csv',
        TRAIN: 'training.csv'
    }[phase]
    submission_csv_file = model_path/csv_name
    submission_csv_file.parent.mkdir(exist_ok=True, parents=True)
    if submission_csv_file.exists():
        print(f"Experience {model_path} already evaluated")
        if not override:
            return submission_csv_file
        else:
            logging.warning(f"Overriding results for experience {model_path}")
    batch_size = configuration[BATCH_SIZE][phase]
    if configuration[PLATFORM]["gpu"].get("Memory", 0)/(1024.**3) < 5.:
        logging.warning("Tiny GPU! Or CPU")
        batch_size = min(batch_size, 8)
    print('loading best model...')
    if forced_batch_size is not None:
        batch_size = forced_batch_size
    best_model_path = sorted(list(model_path.glob("*.pt")))
    if len(best_model_path) == 0:
        best_model_path = sorted(list(backup_folder.glob("*.pt")))
    assert len(best_model_path) > 0, f"No model checkpoint found at {model_path}"
    best_model_path = best_model_path[-1]
    print(best_model_path)
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    if phase == TEST:
        graph_model = model.get_graph_encoder()
        text_model = model.get_text_encoder()
        gt = np.load(DATA_DIR/"token_embedding_dict.npy", allow_pickle=True)[()]
        test_cids_dataset = GraphDataset(root=DATA_DIR, gt=gt, split='test_cids')
        test_text_dataset = TextDataset(
            file_path=DATA_DIR/'test_text.txt', tokenizer=tokenizer)

        test_loader = DataLoader(
            test_cids_dataset, batch_size=batch_size, shuffle=False)

        graph_embeddings = []
        for _, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Test Graph"):
            for output in graph_model(batch.to(device)):
                graph_embeddings.append(output.tolist())

        test_text_loader = TorchDataLoader(
            test_text_dataset, batch_size=batch_size, shuffle=False)
        text_embeddings = []
        for _, batch in tqdm(enumerate(test_text_loader), total=len(test_text_loader), desc="Test Text"):
            for output in text_model(batch['input_ids'].to(device),
                                     attention_mask=batch['attention_mask'].to(device)):
                text_embeddings.append(output.tolist())

        save_embeddings_graph_files = submission_csv_file.parent/'graph_embeddings.npy'
        save_embeddings_text_files = submission_csv_file.parent/'text_embeddings.npy'
        np.save(save_embeddings_graph_files, np.array(graph_embeddings))
        np.save(save_embeddings_text_files, np.array(text_embeddings))
        similarity = cosine_similarity(text_embeddings, graph_embeddings)

        solution = pd.DataFrame(similarity)
        solution['ID'] = solution.index
        solution = solution[['ID'] + [col for col in solution.columns if col != 'ID']]
        solution.to_csv(submission_csv_file, index=False)
        if backup_folder is not None:
            solution.to_csv(backup_folder/csv_name, index=False)
    elif phase == VALIDATION or phase == TRAIN:
        gt = np.load(DATA_DIR/"token_embedding_dict.npy", allow_pickle=True)[()]
        val_dataset = GraphTextDataset(
            root=DATA_DIR,
            gt=gt,
            split={VALIDATION: 'val', TRAIN: 'train'}[phase],
            tokenizer=tokenizer,
            specific_name=configuration[TOKENIZER_NAME])
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        val_loss, lrap_score = eval(
            model, val_loader,
            device=device, score=True,
            desc=phase, max_count=int(np.ceil(3302/batch_size)) if phase == TRAIN else None
        )
        print(f"LRAP score {phase} : {lrap_score:.3%}")
    return submission_csv_file


def evaluate_experience(
    exp: int,
    root_dir: Path = ROOT_DIR, backup_root: Path = None,
    override: bool = False,
    device=None,
    forced_batch_size: int = None,
    phase=TEST
) -> None:
    model, configuration, output_directory, tokenizer, device, backup_folder, _optimizer_state_dict = prepare_experience(
        exp,
        root_dir=root_dir,
        device=device,
        backup_root=backup_root
    )
    submission_csv_file = evaluation(
        model, output_directory, configuration, tokenizer, device, backup_folder=backup_folder,
        override=override,
        phase=phase,
        forced_batch_size=forced_batch_size
    )
    sha1 = get_git_sha1()
    message = f"exp_{exp} sha1: {sha1} config: {configuration}"
    if phase == TEST:
        print(f'kaggle competitions submit -c altegrad-2023-data-challenge -f {submission_csv_file} -m "{message}"')


if __name__ == '__main__':
    parser = get_default_parser()
    parser.add_argument("-b", "--backup-root", type=Path, default=None, help="Backup root folder")
    parser.add_argument("-force", "--force", action="store_true", help="Override results")
    parser.add_argument("-v", "--validation", action="store_true", help="Evaluate on validation set")
    parser.add_argument("-t", "--training", action="store_true", help="Evaluate on training set")
    parser.add_argument("--batch-size", default=None, type=int, help="Forced batch size")
    args = parser.parse_args()
    if args.training:
        phase = TRAIN
    elif args.validation:
        phase = VALIDATION
    else:
        phase = TEST
    for exp in args.exp_list:
        evaluate_experience(
            exp, backup_root=args.backup_root, device=args.device, override=args.force,
            phase=phase,
            forced_batch_size=args.batch_size
        )
