
from loss import contrastive_loss
from tqdm import tqdm
from typing import Optional


def eval(model, val_loader, device='cuda', max_count: Optional[int] = None, score: bool = False):
    model.eval()
    val_loss = 0
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
        current_loss = contrastive_loss(x_graph, x_text)
        val_loss += current_loss.item()

    return val_loss/len(val_loader)
