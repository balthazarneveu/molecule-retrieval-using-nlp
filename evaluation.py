from properties import (
    BATCH_SIZE,
    TEST, DATA_DIR,
    ROOT_DIR
)
from torch_geometric.data import DataLoader
from dataloader import GraphDataset, TextDataset
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader as TorchDataLoader
import pandas as pd
from utils import get_default_parser, prepare_experience


def evaluation(
    model: torch.nn.Module, model_path: Path, configuration: dict, tokenizer, device: str,
    backup_root: Path = None
):
    batch_size = configuration[BATCH_SIZE][TEST]
    print('loading best model...')
    best_model_path = sorted(list(model_path.glob("*.pt")))
    assert len(best_model_path) > 0, "No model checkpoint found at {model_path}"
    best_model_path = best_model_path[-1]
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    graph_model = model.get_graph_encoder()
    text_model = model.get_text_encoder()
    gt = np.load(DATA_DIR/"token_embedding_dict.npy", allow_pickle=True)[()]
    test_cids_dataset = GraphDataset(root=DATA_DIR, gt=gt, split='test_cids')
    test_text_dataset = TextDataset(
        file_path=DATA_DIR/'test_text.txt', tokenizer=tokenizer)

    test_loader = DataLoader(
        test_cids_dataset, batch_size=batch_size, shuffle=False)

    graph_embeddings = []
    for batch in test_loader:
        for output in graph_model(batch.to(device)):
            graph_embeddings.append(output.tolist())

    test_text_loader = TorchDataLoader(
        test_text_dataset, batch_size=batch_size, shuffle=False)
    text_embeddings = []
    for batch in test_text_loader:
        for output in text_model(batch['input_ids'].to(device),
                                 attention_mask=batch['attention_mask'].to(device)):
            text_embeddings.append(output.tolist())

    similarity = cosine_similarity(text_embeddings, graph_embeddings)

    solution = pd.DataFrame(similarity)
    solution['ID'] = solution.index
    solution = solution[['ID'] + [col for col in solution.columns if col != 'ID']]
    solution.to_csv(model_path/'submission.csv', index=False)
    if backup_root is not None:
        solution.to_csv(backup_root/'submission.csv', index=False)


def evaluate_experience(exp: int, root_dir: Path = ROOT_DIR, backup_root: Path = None, device=None) -> None:
    model, configuration, output_directory, tokenizer, device = prepare_experience(
        exp,
        root_dir=root_dir,
        device=device
    )
    evaluation(model, output_directory, configuration, tokenizer, device, backup_root=backup_root)


if __name__ == '__main__':
    parser = get_default_parser()
    parser.add_argument("-b", "--backup-root", type=Path, default=None, help="Backup root folder")
    args = parser.parse_args()
    for exp in args.exp_list:
        evaluate_experience(exp, backup_root=args.backup_root, device=args.device)
