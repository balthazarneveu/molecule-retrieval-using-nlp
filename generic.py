import torch


class GenericModel(torch.nn.Module):
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
