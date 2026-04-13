"""
COS 484 Assignment 4 — Baseline submission (PyTorch eager).

This is a simple reference implementation using PyTorch's built-in functions.
Use this as a starting point and try to beat it!
"""

import torch
import torch.nn.functional as F


def cross_entropy_forward(logits, targets):
    """
    Args:
        logits: (B, V) tensor, bfloat16
        targets: (B,) tensor, int64
    Returns:
        losses: (B,) tensor, float32
    """
    return F.cross_entropy(logits.float(), targets, reduction="none")


def cross_entropy_backward(logits, targets, grad_output):
    """
    Args:
        logits: (B, V) tensor, bfloat16
        targets: (B,) tensor, int64
        grad_output: (B,) tensor, float32
    Returns:
        grad_logits: (B, V) tensor, bfloat16
    """
    probs = torch.softmax(logits.float(), dim=-1)
    probs[torch.arange(logits.shape[0], device=logits.device), targets] -= 1.0
    grad_logits = probs * grad_output.unsqueeze(1)
    return grad_logits.to(logits.dtype)
