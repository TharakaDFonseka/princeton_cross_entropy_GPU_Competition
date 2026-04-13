import torch

# submission_3 — narrow forward: stable log-sum-exp + gather at targets only (no full softmax tensor
# written for the forward pass). Backward matches submission_2 (compiled softmax grad, no probs.clone()).

_COMPILE_MODE = "max-autotune-no-cudagraphs"


@torch.compile(fullgraph=True, dynamic=False, mode=_COMPILE_MODE)
def _compiled_forward(logits, targets):
    x = logits.float()
    m = x.max(dim=1, keepdim=True).values
    shifted = x - m
    lse = shifted.exp().sum(dim=1, keepdim=True).log()
    x_tgt = torch.gather(x, 1, targets.unsqueeze(1)).squeeze(1)
    return -(x_tgt - m.squeeze(1) - lse.squeeze(1))


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
