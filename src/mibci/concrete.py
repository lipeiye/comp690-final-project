"""Learned electrode selection: a concrete (Gumbel-softmax) selector in front of EEGNet.

Strypsteen and Bertrand (2021, "End-to-end learnable EEG channel selection for deep neural
networks with Gumbel-softmax") put k selection neurons before the decoder. Neuron j holds
one logit per electrode; during training it outputs a soft mixture of electrodes drawn
from a Gumbel-softmax whose temperature falls from 10 to 0.1, so each neuron gradually
commits to one electrode while the decoder learns to use it. A penalty discourages two
neurons from taking the same electrode. After training, each neuron's argmax is the
selected electrode (duplicates resolved by the next-best free electrode), and a fresh
EEGNet is trained on the selected subset with the frozen recipe, like every other method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mibci.model import build_eegnet
from mibci.train import set_seed


class ConcreteSelector(nn.Module):
    def __init__(self, n_in: int, k: int):
        super().__init__()
        self.logits = nn.Parameter(0.01 * torch.randn(k, n_in))
        self.temperature = 10.0

    def weights(self) -> torch.Tensor:
        if not self.training:
            return F.one_hot(self.logits.argmax(1), self.logits.shape[1]).float()
        gumbel = -torch.log(-torch.log(torch.rand_like(self.logits).clamp(1e-10, 1 - 1e-10)))
        return torch.softmax((self.logits + gumbel) / self.temperature, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("kc,nct->nkt", self.weights(), x)

    def duplicate_penalty(self) -> torch.Tensor:
        """Probability mass above one on any electrode: zero when every neuron picks a different one."""
        mass = torch.softmax(self.logits / self.temperature, dim=1).sum(0)
        return F.relu(mass - 1).sum()

    def selected(self) -> list[int]:
        """One distinct electrode per neuron, most confident neuron first, in montage order."""
        logits = self.logits.detach().numpy()
        chosen: list[int] = []
        for j in np.argsort(-logits.max(1)):
            free = [c for c in np.argsort(-logits[j]) if c not in chosen]
            chosen.append(int(free[0]))
        return sorted(chosen)


@dataclass(frozen=True)
class SelectorConfig:
    epochs: int = 300
    seed: int = 0
    lr: float = 1e-3
    batch_size: int = 64
    start_temperature: float = 10.0
    end_temperature: float = 0.1
    penalty: float = 1.0


def learn_subset(X: np.ndarray, y: np.ndarray, k: int, cfg: SelectorConfig = SelectorConfig()) -> list[int]:
    """Train selector + EEGNet on (already standardized) training trials; return the k chosen electrodes."""
    set_seed(cfg.seed)
    selector = ConcreteSelector(X.shape[1], k)
    model = nn.Sequential(selector, build_eegnet(k))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)
    X_t, y_t = torch.from_numpy(X), torch.from_numpy(y)
    ratio = cfg.end_temperature / cfg.start_temperature
    for epoch in range(cfg.epochs):
        selector.temperature = cfg.start_temperature * ratio ** (epoch / max(cfg.epochs - 1, 1))
        model.train()
        order = torch.randperm(len(X_t), generator=gen)
        for i in range(0, len(order), cfg.batch_size):
            idx = order[i : i + cfg.batch_size]
            loss = F.cross_entropy(model(X_t[idx]), y_t[idx]) + cfg.penalty * selector.duplicate_penalty()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return selector.selected()
