import torch


def contrastive_loss(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    CE = torch.nn.CrossEntropyLoss()
    logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)


def tempered_contrastive_loss(v1: torch.Tensor, v2: torch.Tensor, temperature: torch.nn.Parameter) -> torch.Tensor:
    CE = torch.nn.CrossEntropyLoss()
    v1n = torch.nn.functional.normalize(v1, dim=1)
    v2n = torch.nn.functional.normalize(v2, dim=1)
    logits = torch.matmul(v1n, torch.transpose(v2n, 0, 1)) * torch.exp(temperature)
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)


def binary_classifier_contrastive_loss(v1, v2):
    BCEL = torch.nn.BCEWithLogitsLoss()
    logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
    eye = torch.eye(v1.shape[0], device=v1.device)
    return BCEL(logits, eye) + BCEL(torch.transpose(logits, 0, 1), eye)
 