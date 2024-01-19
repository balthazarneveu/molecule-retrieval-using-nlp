from loss import contrastive_loss
from tqdm import tqdm
from typing import Optional
from sklearn.metrics import label_ranking_average_precision_score
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch


def eval(model, val_loader, device='cuda', max_count: Optional[int] = None, score: bool = False, desc="Validation"):
    lrap_score = None
    model.eval()
    val_loss = 0
    text_embeddings = []
    graph_embeddings = []

    torch.cuda.empty_cache()  # Just by safety in case training still had some memory allocated
    for batch_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc=desc):
        if max_count is not None and batch_idx > max_count:
            break
        try:
            input_ids = batch.input_ids
            batch.pop('input_ids')
            attention_mask = batch.attention_mask
            batch.pop('attention_mask')
            graph_batch = batch
            x_graph, x_text = model(graph_batch.to(device),
                                    input_ids.to(device),
                                    attention_mask.to(device))
            if x_text.dtype == torch.float16:
                x_graph = x_graph.half()
            current_loss = contrastive_loss(x_graph, x_text)
            val_loss += current_loss.item()
            if score:
                text_embeddings.append(x_text.cpu().detach().numpy())
                graph_embeddings.append(x_graph.cpu().detach().numpy())
        except Exception as exc:
            print(exc)
            pass

    if score:
        # Perform score evaluation on CPU - probably slows things down a bit.
        text_embeddings = np.concatenate(text_embeddings, axis=0)
        graph_embeddings = np.concatenate(graph_embeddings, axis=0)
        all_predictions = cosine_similarity(text_embeddings, graph_embeddings)
        all_true_labels = np.eye(all_predictions.shape[0])

        # LRAP score
        lrap_score = label_ranking_average_precision_score(all_true_labels, all_predictions)

    return val_loss / len(val_loader), lrap_score
