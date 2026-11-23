"""EEGNet bundles and fake quantization for any number of classes (the Lee2019 replication).

mibci.model and mibci.fakequant are pinned by finished studies, and they rebuild a network with
the four classes of BCI IV 2a: load_bundle and deploy_copy call build_eegnet with its default
class count, and fakequant.Folded starts from deploy_copy. The functions here rebuild with the
class count of the network at hand (stored in the bundle); everything else is the pinned code.
"""

from __future__ import annotations

import torch
from torch.nn.utils import parametrize

from mibci import fakequant
from mibci.model import build_eegnet, save_bundle
from mibci.train import ChannelStandardizer


def n_classes(model) -> int:
    return model.final_layer.linearconstraint.out_features


def save(path, model, channels, scaler: ChannelStandardizer, **meta) -> None:
    save_bundle(path, model, channels, scaler, n_classes=n_classes(model), **meta)


def load(path):
    """Like mibci.model.load_bundle, with the bundle's own class count."""
    bundle = torch.load(path, weights_only=True)  # noqa: TID251  load_bundle would rebuild four classes
    model = build_eegnet(len(bundle["channels"]), F2=bundle["F2"], n_classes=bundle["n_classes"])
    model.load_state_dict(bundle["state_dict"])
    scaler = ChannelStandardizer()
    scaler.mean, scaler.std = bundle["mean"].numpy(), bundle["std"].numpy()
    return model.eval(), scaler, bundle


def deploy_copy(model):
    """mibci.model.deploy_copy (max-norm parametrizations baked in) with the model's class count."""
    plain = build_eegnet(model.conv_spatial.kernel_size[0], F2=model.F2, n_classes=n_classes(model))
    plain.load_state_dict(model.state_dict())
    for module in plain.modules():
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
    return plain.eval()


def folded(model) -> fakequant.Folded:
    """fakequant.Folded of a network with any class count.

    Folded builds its copy through the module-level name fakequant.deploy_copy; it is pointed at
    the class-count-aware version above for the one call and restored at once, so the pinned
    module's own behaviour never changes."""
    pinned = fakequant.deploy_copy
    fakequant.deploy_copy = deploy_copy
    try:
        return fakequant.Folded(model)
    finally:
        fakequant.deploy_copy = pinned
