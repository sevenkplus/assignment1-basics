import torch
import einx

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor):
    logits = einx.subtract(
        "... vocab_size, ... -> ... vocab_size",
        inputs,
        einx.logsumexp("... [vocab_size]", inputs)
    )
    losses = -einx.get_at("... [vocab_size], ... -> ...", logits, targets)
    return losses.mean()

def softmax(inputs: torch.Tensor, dim: int):
    x = inputs - inputs.max(dim=dim, keepdim=True).values
    logsumexp = x.exp().sum(dim=dim, keepdim=True).log()
    x -= logsumexp
    return x.exp()
