import torch
import torch.nn.functional as F

# submission_2 — backward avoids probs.clone(): form grad = softmax * g, then
# subtract g at target columns (same math as (softmax - one_hot) * g).

_COMPILE_MODE = "max-autotune-no-cudagraphs"

@torch.compile(fullgraph=True, dynamic=False, mode=_COMPILE_MODE)
def _compiled_forward(logits, targets):
    return F.cross_entropy(logits.float(), targets, reduction="none")

@torch.compile(fullgraph=True, dynamic=False, mode=_COMPILE_MODE)
def _compiled_backward(logits, targets, grad_output):
    x = logits.float()
    m = x.max(dim=1, keepdim=True).values
    shifted = x - m
    lse = shifted.exp().sum(dim=1, keepdim=True).log()
    probs = (shifted - lse).exp()

    rows = torch.arange(targets.shape[0], device=targets.device)
    grad = probs * grad_output[:, None]
    grad[rows, targets] -= grad_output
    return grad.to(torch.bfloat16)

def cross_entropy_forward(logits, targets):
    return _compiled_forward(logits, targets)

def cross_entropy_backward(logits, targets, grad_output):
    return _compiled_backward(logits, targets, grad_output)
