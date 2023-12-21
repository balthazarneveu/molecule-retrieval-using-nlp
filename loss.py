import torch


def contrastive_loss(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    CE = torch.nn.CrossEntropyLoss()
    logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)
