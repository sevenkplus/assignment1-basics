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

def get_batch(dataset, batch_size, context_length, device):
    idx = torch.randint(0, dataset.shape[0]-context_length, (batch_size,))
    input_idx = einx.add(
        "bs, ctl -> bs ctl",
        idx,
        torch.arange(context_length)
    )
    output_idx = einx.add(
        "bs, ctl -> bs ctl",
        idx,
        torch.arange(1, context_length+1)
    )

    inputs = einx.get_at("[n], ... -> ...", dataset, input_idx)
    outputs = einx.get_at("[n], ... -> ...", dataset, output_idx)
    return inputs, outputs

def save_checkpoint(model, optimizer, iteration, out):
    torch.save(
        {"model": model.state_dict(),
         "optim": optimizer.state_dict(),
         "iter": iteration},
        out
    )

def load_checkpoint(src, model, optimizer):
    cp = torch.load(src)
    model.load_state_dict(cp["model"])
    optimizer.load_state_dict(cp["optim"])
    return cp["iter"]
