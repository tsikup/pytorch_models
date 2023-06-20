import torch.nn


def dfs_freeze(model: torch.nn.Module):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)
