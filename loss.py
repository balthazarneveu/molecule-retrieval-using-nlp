import torch
from sklearn.metrics.pairwise import cosine_similarity

def contrastive_loss(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    CE = torch.nn.CrossEntropyLoss()
    logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)

def contrastive_loss_norm(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    CE = torch.nn.CrossEntropyLoss()
    # Ensure gradients are required for further operations
    v1.requires_grad_(True)
    v2.requires_grad_(True)
    cosine_similarity_matrix = cosine_similarity(v1.cpu().detach().numpy(), v2.cpu().detach().numpy())
    print('cosine_similarity_matrix shape:',cosine_similarity_matrix.shape)
    logits = torch.from_numpy(cosine_similarity_matrix).to(v1.device)
    labels = torch.arange(logits.shape[0], device=v1.device)

    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)

def tempered_contrastive_loss(v1: torch.Tensor, v2: torch.Tensor, temperature: torch.nn.Parameter) -> torch.Tensor:
    CE = torch.nn.CrossEntropyLoss()
    logits = torch.matmul(v1, torch.transpose(v2, 0, 1)) * torch.exp(temperature)
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)
