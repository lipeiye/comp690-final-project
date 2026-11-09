"""EEGNet from Braindecode with the settings shared by every condition, and model bundles."""

from __future__ import annotations

import torch
from braindecode.models import EEGNet
from torch.nn.utils import parametrize

from mibci.data import CLASSES, N_TIMES
from mibci.train import ChannelStandardizer

# EEGNet-8,2 (Lawhern et al., 2018). Dropout 0.5 is the paper's within-subject value,
# kernel_length=64 is Braindecode's default (256 ms at 250 Hz), and the dense
# classifier carries the paper's max-norm constraint.
EEGNET_KWARGS = dict(F1=8, D=2, kernel_length=64, drop_prob=0.5, final_layer_with_constraint=True)


def build_eegnet(n_chans: int, F2: int | None = None, n_times: int = N_TIMES,
                 n_classes: int = len(CLASSES)) -> EEGNet:
    """F2 defaults to F1 * D = 16; pruned models pass a smaller F2."""
    return EEGNet(n_chans=n_chans, n_outputs=n_classes, n_times=n_times, F2=F2, **EEGNET_KWARGS)


def save_bundle(path, model: EEGNet, channels, scaler: ChannelStandardizer, **meta) -> None:
    """Weights plus what is needed to run the decoder: input channels and normalization."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "F2": model.F2,
            "channels": [int(c) for c in channels],
            "mean": torch.from_numpy(scaler.mean),
            "std": torch.from_numpy(scaler.std),
            **meta,
        },
        path,
    )


def load_bundle(path) -> tuple[EEGNet, ChannelStandardizer, dict]:
    bundle = torch.load(path, weights_only=True)
    model = build_eegnet(len(bundle["channels"]), F2=bundle["F2"])
    model.load_state_dict(bundle["state_dict"])
    scaler = ChannelStandardizer()
    scaler.mean, scaler.std = bundle["mean"].numpy(), bundle["std"].numpy()
    return model.eval(), scaler, bundle


def deploy_copy(model: EEGNet) -> EEGNet:
    """Eval-mode copy with the max-norm parametrizations baked into plain weights.

    Built as a fresh model rather than a deepcopy: a deepcopy shares the injected
    parametrization class with the original, so removing the parametrization from
    the copy would also strip `weight` from the original.
    """
    plain = build_eegnet(model.conv_spatial.kernel_size[0], F2=model.F2)
    plain.load_state_dict(model.state_dict())
    for module in plain.modules():
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
    return plain.eval()
