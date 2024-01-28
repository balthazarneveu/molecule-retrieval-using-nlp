import torch
from sklearn.metrics.pairwise import cosine_similarity

def contrastive_loss(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    CE = torch.nn.CrossEntropyLoss()
    logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)

def contrastive_loss_norm(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    CE = torch.nn.CrossEntropyLoss()
    cosine_similarity_matrix=torch.nn.functional.cosine_similarity(v1,v2,dim=1)
    labels = torch.arange(cosine_similarity_matrix.shape[0], device=v1.device)
    return CE(cosine_similarity_matrix, labels) + CE(cosine_similarity_matrix.T, labels)

def tempered_contrastive_loss(v1: torch.Tensor, v2: torch.Tensor, temperature: torch.nn.Parameter) -> torch.Tensor:
    CE = torch.nn.CrossEntropyLoss()
    logits = torch.matmul(v1, torch.transpose(v2, 0, 1)) * torch.exp(temperature)
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)
