import torch
from sklearn.metrics.pairwise import cosine_similarity

def contrastive_loss(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    CE = torch.nn.CrossEntropyLoss()
    logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)

def contrastive_loss_norm(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    CE = torch.nn.CrossEntropyLoss()
    logits = torch.nn.functional.cosine_similarity(v1.unsqueeze(1), v2.unsqueeze(0), dim=2)
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits,0,1), labels)

def tempered_contrastive_loss(v1: torch.Tensor, v2: torch.Tensor, temperature: torch.nn.Parameter) -> torch.Tensor:
    CE = torch.nn.CrossEntropyLoss()
    logits = torch.matmul(v1, torch.transpose(v2, 0, 1)) * torch.exp(temperature)
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)
