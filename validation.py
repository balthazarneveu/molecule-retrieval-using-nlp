from loss import contrastive_loss
from tqdm import tqdm
from typing import Optional
from sklearn.metrics import label_ranking_average_precision_score
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


def eval(model, val_loader, device='cuda', max_count: Optional[int] = None, score: bool = False):
    lrap_score = None
    model.eval()
    val_loss = 0
    text_embeddings = []
    graph_embeddings = []
    for batch_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation"):
        if max_count is not None and batch_idx > max_count:
            continue

        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch

        x_graph, x_text = model(graph_batch.to(device),
                                input_ids.to(device),
                                attention_mask.to(device))

        if score:
            text_embeddings.append(x_text.cpu().detach().numpy())
            graph_embeddings.append(x_graph.cpu().detach().numpy())

        current_loss = contrastive_loss(x_graph, x_text)
        val_loss += current_loss.item()

    if score:
        text_embeddings = np.concatenate(text_embeddings, axis=0)
        graph_embeddings = np.concatenate(graph_embeddings, axis=0)
        all_predictions = cosine_similarity(text_embeddings, graph_embeddings)
        all_true_labels = np.eye(all_predictions.shape[0])

        # LRAP score
        lrap_score = label_ranking_average_precision_score(all_true_labels, all_predictions)

    return val_loss / len(val_loader), lrap_score
