from collections.abc import Callable, Iterable
from typing import Optional

import torch
import torch.optim as optim
import math

class SGD(optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t+1) * grad
                state["t"] = t+1
        return loss

class AdamW(optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "lm": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            eps = group["eps"]
            lm = group["lm"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                m = state.get("m", torch.zeros_like(grad))
                v = state.get("v", torch.zeros_like(grad))
                t = state.get("t", 1)

                m = beta1*m + (1-beta1)*grad
                v = beta2*v + (1-beta2)*grad**2
                lr_t = lr * math.sqrt(1-beta2**t)  / (1-beta1**t)

                p.data -= lr_t * m / (v.sqrt() + eps)
                p.data *= (1-lr*lm)

                state["m"] = m
                state["v"] = v
                state["t"] = t+1
        return loss

def get_lr_cosine_schedule(t: int, lr_max: float, lr_min: float, tw: int, tc: int):
    if t < tw:
        return lr_max*t/tw
    if tw <= t <= tc:
        return lr_min + 1/2*(1+math.cos((t-tw)/(tc-tw)*math.pi))*(lr_max-lr_min)
    return lr_min

def gradient_clipping(params, max_l2, eps=1e-6):
    s = 0
    for p in params:
        if p.grad is not None:
            s += (p.grad**2).sum()
    if s < max_l2: return
    f = max_l2 / (math.sqrt(s) + eps)
    for p in params:
        if p.grad is not None:
            p.grad *= f
