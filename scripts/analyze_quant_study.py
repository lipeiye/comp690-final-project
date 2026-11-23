"""Tables and figures for the bit-width study of scripts/quant_study.py (post-test, exploratory).

Reads the study's own records and nothing else; no EEG is loaded. Under RESULTS_DIR (MIBCI_RESULTS_DIR):
  supplementary/quant_study/protocol.json   the plan: jobs, bit widths, mixed settings, QAT variants
  supplementary/quant_study/ptq/*.json      post-training quantization (PTQ) of every archived final model
  supplementary/quant_study/qat/*.json      quantization-aware fine-tuning (QAT); skipped with --ptq-only
  frozen_config.json                        the frozen bootstrap plan
  supplementary/int8_study/costs.csv        MACs of the layer-swapped graph, when that study has run

Writes to RESULTS_DIR/supplementary/quant_study/:
  ptq_grid.csv      accuracy at every weight x activation bit width and mixed setting, per condition and
                    pooled over all conditions, against float/float of the same models
  sensitivity.csv   accuracy change when one weight layer or one activation tensor alone is rounded,
                    on session 1 (the training trials) and on session 2
  mixed.csv         the mixed settings against float/float, 8/8 and 4/4
  qat.csv           accuracy after fine-tuning, against the matched floating-point control (fp32-ft)
                    and against the same rounding without fine-tuning
  costs.csv         per architecture and setting: weight bytes, peak activation bytes, MACs,
                    bit-operations and arithmetic energy, for the original and the layer-swapped graph
  fig_ptq_heatmap.png, fig_sensitivity.png, fig_cost_accuracy.png, summary.md, analysis.json

Statistics follow the frozen plan: seeds (and, in pooled rows, conditions) are averaged within each
participant, then the nine participants are summarized; intervals are the participant-level percentile
bootstrap (10,000 resamples, 95 %, seed 690).

Boundaries, repeated in summary.md: bit widths below 8 are simulated (integer grids evaluated in
floating point) and no low-bit kernel was run or timed; session-2 numbers are exploratory, because the
study was planned after the main study's test results were known; the precision rule is read from
session-1 sensitivity only; the energy figure counts arithmetic only (Horowitz, ISSCC 2014, 45 nm), not
memory traffic, and is an order-of-magnitude estimate, not a measurement.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy import stats

from mibci import costs, fakequant
from mibci.artifacts import sha256, write_json
from mibci.conditions import parse_condition
from mibci.layout import frozen_config
from mibci.model import EEGNET_KWARGS, build_eegnet
from mibci.paths import RESULTS_DIR, display
from mibci.stats import bootstrap_ci

OUT = RESULTS_DIR / "supplementary/quant_study"
INT8_COSTS = RESULTS_DIR / "supplementary/int8_study/costs.csv"
INT8_COLUMNS = (
    "electrodes",
    "F2",
    "original_macs",
    "original_params",
    "spatial-first_macs",
    "spatial-first_peak_activation_elems",
)
PTQ_POOLED, QAT_POOLED = "all conditions", "all QAT conditions"
FLOAT, W8A8, W4A4 = "float/float", "8/8", "4/4"
CONTROL = "fp32-ft"  # the QAT variant fine-tuned without rounding: the matched control
SHOWN = ("full-22", "relevance-16-prune50")  # all 22 electrodes, and the frozen finalist
KEY_SETTINGS = ("16/16", "8/8", "6/6", "5/5", "4/8", "4/4", "3/8", "2/8")
COST_SETTINGS = (FLOAT, "16/16", "8/8", "4/8", "4/4", "2/8")
MATERIAL_PP = 2.0  # the frozen plan counts participants who lose more than 2 points
FLOAT_BITS = 32  # how fakequant.bit_operations counts a floating-point operand
BASE_F2 = EEGNET_KWARGS["F1"] * EEGNET_KWARGS["D"]

# Energy per operation at 45 nm, in picojoules (Horowitz, "Computing's energy problem", ISSCC 2014).
INT8_MULTIPLY_PJ = 0.2  # a w-bit x a-bit multiply is taken as 0.2 * (w/8) * (a/8): 3.2 at 32 bits, 3.1 reported
INT32_ADD_PJ = 0.1  # the accumulate of an integer MAC, in 32 bits
FLOAT32_MULTIPLY_PJ, FLOAT32_ADD_PJ = 3.7, 0.9

# costs.layer_costs row -> (tensor the layer reads, tensor it writes), named as the taps of fakequant.Folded
LAYER_TAPS = {
    "conv_temporal": ("input", "conv_t"),
    "conv_spatial": ("conv_t", "conv_s"),
    "elu_1": ("conv_s", "elu1"),
    "pool_1": ("elu1", "pool1"),
    "conv_separable_depth": ("pool1", "conv_d"),
    "conv_separable_point": ("conv_d", "conv_p"),
    "elu_2": ("conv_p", "elu2"),
    "pool_2": ("elu2", "pool2"),
    "final_layer.linearconstraint": ("pool2", "logits"),
}
PLAIN = {
    "input": "input EEG",
    "conv_t": "temporal convolution",
    "conv_s": "spatial convolution",
    "elu1": "first ELU",
    "pool1": "first pooling",
    "conv_d": "depthwise convolution",
    "conv_p": "pointwise convolution",
    "elu2": "second ELU",
    "pool2": "second pooling",
    "logits": "logits",
    "fc": "classifier",
}

# Chart colors: a colorblind-checked categorical pair, gray for context, red-gray-blue for signed changes
SURFACE, INK, SECONDARY, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, RED, NEUTRAL = "#2a78d6", "#eb6834", "#e34948", "#f0efec"
MIXED_MARKERS = "Ds^v"  # one shape per mixed setting in the cost figure
STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": SECONDARY,
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": SECONDARY,
    "ytick.labelcolor": SECONDARY,
    "text.color": INK,
    "font.size": 9,
    "legend.frameon": False,
}


# ---------------------------------------------------------------- plan and records


def load_plan() -> dict:
    path = OUT / "protocol.json"
    if not path.exists():
        raise SystemExit(f"No plan at {display(path)}: run scripts/quant_study.py --stage plan, then --stage run")
    return json.loads(path.read_text())


def settings_of(plan: dict) -> dict[str, tuple]:
    """Every PTQ setting in the plan's order: 'W/A' or a mixed key -> (weight bits, activation bits).

    A width is an int, None (floating point), or a mapping from layer or tensor name to bits."""

    def width(label: str) -> int | None:
        return None if label == "float" else int(label)

    grid = {f"{w}/{a}": (width(w), width(a)) for w in plan["weight_bits"] for a in plan["activation_bits"]}
    return grid | {key: (spec["weights"], spec["activations"]) for key, spec in plan["mixed"].items()}


def uniform(spec) -> bool:
    return not isinstance(spec, Mapping)


def bits_label(spec) -> str:
    """'float', a width such as '8', or for a per-name mapping e.g. 'input, conv_t 8; rest 4'."""
    if spec is None:
        return "float"
    if uniform(spec):
        return str(spec)
    common = Counter(spec.values()).most_common(1)[0][0]
    other: dict[int, list[str]] = {}
    for name, bits in spec.items():
        if bits != common:
            other.setdefault(bits, []).append(name)
    parts = [f"{', '.join(names)} {bits}" for bits, names in sorted(other.items(), reverse=True)]
    return "; ".join([*parts, f"rest {common}"])


def read_records(plan: dict, kind: str) -> list[dict]:
    """Every record of one kind in the plan, checked against its job and against this plan's hash."""
    jobs = [job for job in plan["jobs"] if job["kind"] == kind]
    paths = [OUT / kind / f"{job['name']}.json" for job in jobs]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        hint = " or pass --ptq-only" if kind == "qat" else ""
        raise SystemExit(
            f"{len(missing)} of {len(paths)} {kind} records are missing (first: {display(missing[0])}). "
            f"Finish scripts/quant_study.py --stage run --kind {kind}{hint}."
        )
    digest = sha256(OUT / "protocol.json")
    records = []
    for job, path in zip(jobs, paths, strict=True):
        record = json.loads(path.read_text())
        if any(record.get(key) != value for key, value in job.items()) or record.get("protocol_sha256") != digest:
            raise SystemExit(f"{display(path)} does not belong to this plan: its job fields or protocol hash differ")
        records.append(record)
    return records


def split_tensor(tensor: str) -> tuple[str, str, int]:
    """'weights:conv_s@2' -> ('weights', 'conv_s', 2)."""
    kind, rest = tensor.split(":")
    name, bits = rest.split("@")
    return kind, name, int(bits)


def tensor_order(tensor: str) -> tuple:
    """Weights at 4 bits, weights at 2 bits, then activations; each in network order."""
    kind, name, bits = split_tensor(tensor)
    names = fakequant.WEIGHTED if kind == "weights" else fakequant.POINTS
    return kind != "weights", -bits, names.index(name)


def plain(kind: str, name: str) -> str:
    """'spatial convolution' for a weight layer, 'spatial convolution output' for an activation tensor."""
    if kind == "weights" or name in ("input", "logits"):
        return PLAIN[name]
    return f"{PLAIN[name]} output"


def tensor_name(tensor: str) -> str:
    """'weights:conv_s@2' -> 'spatial convolution weights, 2 bits'."""
    kind, name, bits = split_tensor(tensor)
    what = f"{PLAIN[name]} weights" if kind == "weights" else plain(kind, name)
    return f"{what}, {bits} bits"


def ptq_frames(records: list[dict], settings: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Long tables: grid accuracy per model and setting, sensitivity change per model, simulator check per model."""
    grid, sens, sim = [], [], []
    for r in records:
        model = {"subject": r["subject"], "seed": r["seed"], "condition": r["condition"]}
        for setting in settings:
            cell = r["grid"][setting]
            grid.append({**model, "setting": setting, "acc": cell["primary_acc"], "acc_all": cell["secondary_acc"]})
        s = r["sensitivity"]
        # session 1: the model's own training trials; session 2: the artifact-free trials (primary analysis)
        for session, reference in (
            ("session1", s["session1_float_acc"]),
            ("session2", r["grid"][FLOAT]["primary_acc"]),
        ):
            sens.extend(
                {**model, "session": session, "tensor": t, "change": acc - reference} for t, acc in s[session].items()
            )
        agreement, archived = r["simulated_w8a8_agreement_with_archived"], r["archived_primary_acc"]
        sim.append(
            {
                **model,
                "agree_int8": agreement["int8"],
                "agree_fp32": agreement["fp32"],
                "sim_w8a8": r["grid"][W8A8]["primary_acc"],
                "ort_int8": archived["int8"],
                "sim_float": r["grid"][FLOAT]["primary_acc"],
                "ort_fp32": archived["fp32"],
            }
        )
    return pd.DataFrame(grid), pd.DataFrame(sens), pd.DataFrame(sim)


def qat_frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject": r["subject"],
                "seed": r["seed"],
                "condition": r["condition"],
                "variant": r["variant"],
                "before": r["before"]["primary"]["acc"],
                "after": r["after"]["primary"]["acc"],
                "after_all": r["after"]["secondary"]["acc"],
                "train_after": r["train_after"]["acc"],
            }
            for r in records
        ]
    )


# ---------------------------------------------------------------- participant-level statistics


def with_pooled(df: pd.DataFrame, keys: list[str], label: str) -> pd.DataFrame:
    """Adds rows for condition `label`: each participant and seed averaged over the conditions first."""
    pooled = df.groupby([*keys, "subject", "seed"], as_index=False).mean(numeric_only=True)
    pooled["condition"] = label
    return pd.concat([pooled, df], ignore_index=True)


def participant_means(df: pd.DataFrame, keys: list[str], value: str) -> pd.DataFrame:
    """Rows `keys`, one column per participant, seeds averaged within participant; fractions become %."""
    wide = df.groupby([*keys, "subject"])[value].mean().unstack("subject") * 100
    if wide.isna().to_numpy().any():
        raise SystemExit(f"Incomplete records: some participant lacks a value of {value} for some {keys}")
    return wide


def material(change):
    """True where a change in pp is a loss of more than MATERIAL_PP points. Whole trials can make a loss of
    exactly 2 points; the tolerance keeps float rounding from counting it as more."""
    return change < -(MATERIAL_PP + 1e-9)


def difference(a, b, boot: dict, prefix: str = "") -> dict:
    """a minus b per participant (pp): mean, bootstrap interval, worst participant, participants losing > 2 points."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    low, high = bootstrap_ci(d, boot)
    return {
        f"{prefix}diff_pp": float(d.mean()),
        f"{prefix}ci_low_pp": low,
        f"{prefix}ci_high_pp": high,
        f"{prefix}worst_participant_pp": float(d.min()),
        f"{prefix}n_losing_over_2pp": int(material(d).sum()),
    }


def condition_order(conditions, pooled: str) -> list[str]:
    """The pooled row first, then the conditions by electrode count (most first) and name."""
    rest = set(conditions) - {pooled}
    return [pooled, *sorted(rest, key=lambda c: (-parse_condition(c)["electrodes"], c))]


def ptq_accuracy(grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per participant (columns) and (condition, setting) rows: primary and all-trial accuracy in %."""
    grid = with_pooled(grid, ["setting"], PTQ_POOLED)
    keys = ["condition", "setting"]
    return participant_means(grid, keys, "acc"), participant_means(grid, keys, "acc_all")


def ptq_grid(acc: pd.DataFrame, acc_all: pd.DataFrame, settings: dict, boot: dict) -> pd.DataFrame:
    rows = []
    for condition in condition_order(acc.index.get_level_values("condition"), PTQ_POOLED):
        reference = acc.loc[(condition, FLOAT)].to_numpy()
        for setting, (wb, ab) in settings.items():
            values = acc.loc[(condition, setting)].to_numpy()
            rows.append(
                {
                    "condition": condition,
                    "setting": setting,
                    "weight_bits": bits_label(wb),
                    "activation_bits": bits_label(ab),
                    "acc_mean": values.mean(),
                    **difference(values, reference, boot, "vs_float_"),
                    "acc_all_trials": acc_all.loc[(condition, setting)].mean(),
                }
            )
    return pd.DataFrame(rows)


def simulator_check(sim: pd.DataFrame, boot: dict) -> dict:
    """Does the simulated W8A8 network behave like the archived ONNX Runtime INT8 graphs?"""
    per = sim.groupby("subject")[["sim_w8a8", "ort_int8"]].mean() * 100
    gap = (sim.sim_w8a8 - sim.ort_int8).abs() * 100
    return {
        "models": len(sim),
        "agreement_with_archived_int8_pct_mean": 100 * float(sim.agree_int8.mean()),
        "agreement_with_archived_int8_pct_min": 100 * float(sim.agree_int8.min()),
        "agreement_with_archived_fp32_pct_mean": 100 * float(sim.agree_fp32.mean()),
        "abs_acc_diff_sim_w8a8_vs_ort_int8_pp_mean": float(gap.mean()),
        "abs_acc_diff_sim_w8a8_vs_ort_int8_pp_max": float(gap.max()),
        "abs_acc_diff_float_vs_archived_fp32_pp_mean": 100 * float((sim.sim_float - sim.ort_fp32).abs().mean()),
        "sim_w8a8_minus_ort_int8": difference(per.sim_w8a8, per.ort_int8, boot),
    }


def sensitivity_table(sens: pd.DataFrame, boot: dict) -> pd.DataFrame:
    tensors = sorted(dict.fromkeys(sens.tensor), key=tensor_order)
    sens = with_pooled(sens, ["session", "tensor"], PTQ_POOLED)
    change = participant_means(sens, ["condition", "session", "tensor"], "change")
    rows = []
    for condition in condition_order(sens.condition, PTQ_POOLED):
        for session in ("session1", "session2"):
            for tensor in tensors:
                kind, name, bits = split_tensor(tensor)
                values = change.loc[(condition, session, tensor)].to_numpy()
                rows.append(
                    {
                        "condition": condition,
                        "session": session,
                        "tensor": tensor,
                        "kind": kind,
                        "name": name,
                        "bits": bits,
                        "description": tensor_name(tensor),
                        **difference(values, np.zeros_like(values), boot),
                    }
                )
    return pd.DataFrame(rows)


def planned_wider(plan: dict) -> dict[str, list[str]]:
    """Names that the plan's mixed settings keep wider than the rest (early activations, spatial weights)."""
    wider: dict[str, list[str]] = {"weights": [], "activations": []}
    for spec in plan["mixed"].values():
        for kind, names in wider.items():
            bits = spec[kind]
            if not uniform(bits):
                names.extend(n for n, b in bits.items() if b > min(bits.values()) and n not in names)
    return wider


def precision_rule(table: pd.DataFrame, plan: dict) -> dict:
    """From session 1 only: the tensors whose rounding alone costs more than MATERIAL_PP points on average."""
    s1 = table[(table.condition == PTQ_POOLED) & (table.session == "session1")]
    flagged = s1[material(s1.diff_pp)]

    def names(kind: str, bits: int) -> list[str]:
        return flagged[(flagged.kind == kind) & (flagged.bits == bits)].name.tolist()

    pooled = table[table.condition == PTQ_POOLED].pivot(index="tensor", columns="session", values="diff_pp")
    rho = stats.spearmanr(pooled.session1, pooled.session2)
    planned = planned_wider(plan)
    return {
        "threshold_pp": MATERIAL_PP,
        "activations_need_more_than_4_bits": names("activations", 4),
        "weights_need_more_than_4_bits": names("weights", 4),
        "weights_need_more_than_2_bits": names("weights", 2),
        "planned_activations_at_8_bits": planned["activations"],
        "planned_weights_at_4_bits": planned["weights"],
        "spearman_session1_vs_session2": {"rho": float(rho.statistic), "p": float(rho.pvalue), "n": len(pooled)},
    }


def mixed_table(acc: pd.DataFrame, plan: dict, boot: dict) -> pd.DataFrame:
    rows = []
    for condition in condition_order(acc.index.get_level_values("condition"), PTQ_POOLED):
        for setting in plan["mixed"]:
            values = acc.loc[(condition, setting)].to_numpy()
            for reference in (FLOAT, W8A8, W4A4):
                base = acc.loc[(condition, reference)].to_numpy()
                rows.append(
                    {
                        "condition": condition,
                        "setting": setting,
                        "reference": reference,
                        "acc_mean": values.mean(),
                        "reference_acc_mean": base.mean(),
                        **difference(values, base, boot),
                    }
                )
    return pd.DataFrame(rows)


def qat_table(qat: pd.DataFrame, plan: dict, boot: dict) -> pd.DataFrame:
    qat = with_pooled(qat, ["variant"], QAT_POOLED)
    keys = ["condition", "variant"]
    before, after = participant_means(qat, keys, "before"), participant_means(qat, keys, "after")
    after_all, train = participant_means(qat, keys, "after_all"), participant_means(qat, keys, "train_after")
    rows = []
    for condition in condition_order(qat.condition, QAT_POOLED):
        control = after.loc[(condition, CONTROL)].to_numpy()
        for variant, spec in plan["qat"]["variants"].items():
            tuned, start = after.loc[(condition, variant)].to_numpy(), before.loc[(condition, variant)].to_numpy()
            rows.append(
                {
                    "condition": condition,
                    "variant": variant,
                    "weight_bits": bits_label(spec["weights"]),
                    "activation_bits": bits_label(spec["activations"]),
                    "acc_before": start.mean(),
                    "acc_after": tuned.mean(),
                    **difference(tuned, control, boot, "vs_fp32ft_"),
                    **difference(tuned, start, boot, "vs_before_"),
                    "acc_all_trials_after": after_all.loc[(condition, variant)].mean(),
                    "train_acc_after": train.loc[(condition, variant)].mean(),
                }
            )
    return pd.DataFrame(rows)


def families(plan: dict) -> dict[str, list[str]]:
    """Three lines through the grid of uniform settings, widest first."""
    weights, activations = plan["weight_bits"], plan["activation_bits"]
    return {
        "same bits for weights and activations": [f"{b}/{b}" for b in weights if b in activations and b != "float"],
        "weight bits, with 8-bit activations": [f"{b}/8" for b in weights if b != "float"],
        "activation bits, with floating-point weights": [f"float/{b}" for b in activations if b != "float"],
    }


def narrowest_safe(table: pd.DataFrame, plan: dict) -> dict[str, dict[str, str | None]]:
    """Per condition and family: the last setting, going from wide to narrow, before the first one at which
    some participant loses more than MATERIAL_PP points against float/float (None if the widest already does)."""
    out = {}
    for condition, rows in table.groupby("condition", sort=False):
        lost = rows.set_index("setting").vs_float_n_losing_over_2pp
        out[condition] = {}
        for family, settings in families(plan).items():
            safe = None
            for setting in settings:
                if lost[setting] > 0:
                    break
                safe = setting
            out[condition][family] = safe
    return out


# ---------------------------------------------------------------- costs


def architecture(condition: str) -> tuple[int, int]:
    """(electrodes, F2) of a condition; pruning drops filters as mibci.prune.prune_pointwise does."""
    parsed = parse_condition(condition)
    return parsed["electrodes"], BASE_F2 - round(parsed["pruned"] / 100 * BASE_F2)


def architecture_rows(cost: pd.DataFrame, condition: str) -> pd.DataFrame:
    """The costs.csv rows of the architecture that `condition` uses, indexed by setting."""
    electrodes, f2 = architecture(condition)
    return cost[(cost.electrodes == electrodes) & (cost["F2"] == f2)].set_index("setting")


def layer_swap_costs() -> dict[tuple[int, int], pd.Series]:
    """The INT8 study's analytical costs per architecture, or nothing when that study has not run."""
    if not INT8_COSTS.exists():
        return {}
    table = pd.read_csv(INT8_COSTS)
    missing = [c for c in INT8_COLUMNS if c not in table.columns]
    if missing:
        raise SystemExit(f"{display(INT8_COSTS)} lacks the columns {missing}; expected the INT8 study's cost table")
    return {(int(row.electrodes), int(row.F2)): row for _, row in table.iterrows()}


def mac_energy_pj(weight_bits: int, activation_bits: int) -> float:
    """One multiply-accumulate; 32 bits marks a floating-point operand, as in fakequant.bit_operations."""
    if FLOAT_BITS in (weight_bits, activation_bits):
        return FLOAT32_MULTIPLY_PJ + FLOAT32_ADD_PJ
    return INT8_MULTIPLY_PJ * (weight_bits / 8) * (activation_bits / 8) + INT32_ADD_PJ


def peak_bytes(layers: list[dict], activation_bits) -> float:
    """The largest input + output pair of one layer (as in costs.py), each tensor at its own width."""

    def size(elems: int, tap: str) -> float:
        return elems * (fakequant.bits_of(activation_bits, tap) or FLOAT_BITS) / 8

    return max(
        size(r["in_elems"], LAYER_TAPS[r["layer"]][0]) + size(r["out_elems"], LAYER_TAPS[r["layer"]][1]) for r in layers
    )


def layer_swapped(row, wb, ab) -> dict:
    """The layer-swapped graph. MACs come from the INT8 study; bit-operations, energy and peak activation bytes
    are given only where all weights share one width and all activations another, because its layers are not
    the simulated ones."""
    out = dict.fromkeys(
        (
            "spatial_first_macs",
            "spatial_first_bit_operations",
            "spatial_first_mac_energy_uj",
            "spatial_first_peak_activation_bytes",
        ),
        np.nan,
    )
    if row is None:
        return out
    macs = int(row["spatial-first_macs"])
    out["spatial_first_macs"] = macs
    if uniform(wb) and uniform(ab):
        w, a = wb or FLOAT_BITS, ab or FLOAT_BITS
        out["spatial_first_bit_operations"] = macs * w * a
        out["spatial_first_mac_energy_uj"] = macs * mac_energy_pj(w, a) / 1e6
        out["spatial_first_peak_activation_bytes"] = int(row["spatial-first_peak_activation_elems"]) * a / 8
    return out


def architecture_costs(electrodes: int, f2: int, conditions: list[str], settings: dict, swapped) -> list[dict]:
    """Analytical costs of one architecture at every setting, for one 2-s window at batch 1."""
    model = build_eegnet(electrodes, F2=f2)
    folded = fakequant.Folded(model)  # BatchNorm folded, as exported: the values a deployment stores
    layers = {name: getattr(folded, name) for name in fakequant.WEIGHTED}
    weights = {name: layer.weight.numel() for name, layer in layers.items()}
    channels = {name: layer.weight.shape[0] for name, layer in layers.items()}
    biases = sum(layer.bias.numel() for layer in layers.values() if layer.bias is not None)
    summary, rows = costs.cost_summary(model), costs.layer_costs(model)
    if swapped is not None:
        archived = int(swapped["original_macs"]), int(swapped["original_params"])
        if archived != (summary["macs"], sum(weights.values()) + biases):
            raise SystemExit(f"{display(INT8_COSTS)} disagrees with the {electrodes}-electrode, F2={f2} model here")
    out = []
    for setting, (wb, ab) in settings.items():
        ops = fakequant.bit_operations(model, wb, ab)
        widths = {name: fakequant.bits_of(wb, name) for name in weights}
        energy_pj = sum(r["macs"] * mac_energy_pj(r["weight_bits"], r["activation_bits"]) for r in ops["layers"])
        out.append(
            {
                "electrodes": electrodes,
                "F2": f2,
                "conditions": " ".join(conditions),
                "setting": setting,
                "weight_bits": bits_label(wb),
                "activation_bits": bits_label(ab),
                "weights": sum(weights.values()),
                "biases": biases,
                "weight_bytes": sum(n * (widths[name] or FLOAT_BITS) for name, n in weights.items()) / 8 + 4 * biases,
                "weight_scale_bytes": 4 * sum(channels[name] for name, bits in widths.items() if bits is not None),
                "peak_activation_bytes": peak_bytes(rows, ab),
                "macs": summary["macs"],
                "bit_operations": ops["bops"],
                "mac_energy_uj": energy_pj / 1e6,
                **layer_swapped(swapped, wb, ab),
            }
        )
    return out


def cost_table(conditions, settings: dict) -> pd.DataFrame:
    swapped = layer_swap_costs()
    groups: dict[tuple[int, int], list[str]] = {}
    for condition in condition_order(conditions, PTQ_POOLED)[1:]:
        groups.setdefault(architecture(condition), []).append(condition)
    rows = []
    for (electrodes, f2), members in groups.items():
        rows.extend(architecture_costs(electrodes, f2, members, settings, swapped.get((electrodes, f2))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures


def figure_heatmap(table: pd.DataFrame, plan: dict) -> None:
    weights, activations = plan["weight_bits"], plan["activation_bits"]
    matrices = {}
    for condition in SHOWN:
        diff = table[table.condition == condition].set_index("setting").vs_float_diff_pp
        matrices[condition] = np.array([[diff[f"{w}/{a}"] for a in activations] for w in weights])
    limit = max(1.0, *(np.abs(m).max() for m in matrices.values()))
    colors = LinearSegmentedColormap.from_list("loss_gain", [RED, NEUTRAL, BLUE])
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(SHOWN), figsize=(11.5, 4.9), constrained_layout=True)
        for ax, (condition, m) in zip(axes, matrices.items(), strict=True):
            image = ax.imshow(m, cmap=colors, norm=Normalize(-limit, limit), aspect="auto")
            for i, j in np.ndindex(m.shape):
                ax.text(j, i, f"{m[i, j]:+.1f}", ha="center", va="center", fontsize=8, color=INK)
            ax.set(
                xticks=range(len(activations)),
                xticklabels=activations,
                yticks=range(len(weights)),
                yticklabels=weights,
                xlabel="activation bits",
                ylabel="weight bits",
                title=f"{condition}: accuracy minus float/float (pp)",
            )
            ax.grid(False)
            ax.spines[:].set_visible(False)
        fig.colorbar(image, ax=axes, shrink=0.85, label="change in held-out-day accuracy (pp)")
        fig.savefig(OUT / "fig_ptq_heatmap.png", dpi=150)
        plt.close(fig)


def figure_sensitivity(table: pd.DataFrame) -> None:
    pooled = table[table.condition == PTQ_POOLED]
    tensors = list(dict.fromkeys(pooled.tensor))
    y = np.arange(len(tensors))
    sessions = (
        ("session1", BLUE, "session 1, training trials (the rule is read here)"),
        ("session2", ORANGE, "session 2, held-out day (exploratory)"),
    )
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8.6, 7.6))
        for offset, (session, color, label) in zip((-0.19, 0.19), sessions, strict=True):
            rows = pooled[pooled.session == session].set_index("tensor").loc[tensors]
            spread = [np.clip(rows.diff_pp - rows.ci_low_pp, 0, None), np.clip(rows.ci_high_pp - rows.diff_pp, 0, None)]
            ax.barh(
                y + offset,
                rows.diff_pp,
                height=0.36,
                color=color,
                label=label,
                xerr=spread,
                error_kw={"elinewidth": 0.8, "ecolor": SECONDARY, "capsize": 0},
            )
        ax.axvline(0, color="#c3c2b7", lw=1)
        ax.axvline(-MATERIAL_PP, color=SECONDARY, lw=1, ls="--", label=f"{MATERIAL_PP:g}-point threshold")
        groups = [tensor_order(t)[:2] for t in tensors]
        for i in range(1, len(tensors)):
            if groups[i] != groups[i - 1]:
                ax.axhline(i - 0.5, color=GRID, lw=1)
        ax.set_xscale("symlog", linthresh=1)
        low = min(pooled.ci_low_pp.min(), -MATERIAL_PP)
        ticks = [
            t
            for t in (-100, -50, -20, -10, -5, -2, -1, 0, 1, 2, 5)
            if low * 1.3 <= t <= max(1, pooled.ci_high_pp.max())
        ]
        ax.set_xticks(ticks, [f"{t:+d}" if t else "0" for t in ticks])
        ax.set(yticks=y, yticklabels=[tensor_name(t) for t in tensors], ylim=(len(tensors) - 0.5, -0.5))
        ax.set_xlabel("accuracy change when only this tensor is rounded (pp; linear within ±1, logarithmic beyond)")
        ax.set_title("Rounding one tensor at a time (all conditions; mean and 95% interval)", loc="left")
        ax.grid(axis="y", visible=False)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(OUT / "fig_sensitivity.png", dpi=150)
        plt.close(fig)


def figure_cost_accuracy(table: pd.DataFrame, cost: pd.DataFrame, plan: dict, settings: dict) -> None:
    diagonal = [FLOAT, *families(plan)["same bits for weights and activations"]]
    others = [s for s, (wb, ab) in settings.items() if uniform(wb) and uniform(ab) and s not in diagonal]
    mixed = list(plan["mixed"])
    axes_spec = (
        ("weight_bytes", "weight storage (bytes, log scale)"),
        ("bit_operations", "bit-operations (log scale)"),
    )
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(SHOWN), 2, figsize=(11.5, 8.4), sharey="row")
        for row_axes, condition in zip(axes, SHOWN, strict=True):
            acc = table[table.condition == condition].set_index("setting").acc_mean
            arch = architecture_rows(cost, condition)
            for ax, (column, xlabel) in zip(row_axes, axes_spec, strict=True):
                x = arch[column]
                ax.scatter(x[others], acc[others], s=14, color=MUTED, alpha=0.45, lw=0, label="other uniform settings")
                ax.plot(
                    x[diagonal],
                    acc[diagonal],
                    color=BLUE,
                    lw=2,
                    marker="o",
                    ms=5,
                    label="same bits for weights and activations",
                )
                for s in diagonal:
                    ax.annotate(
                        s.split("/")[0],
                        (x[s], acc[s]),
                        xytext=(5, -11),
                        textcoords="offset points",
                        fontsize=7,
                        color=SECONDARY,
                    )
                for s, marker in zip(mixed, MIXED_MARKERS[: len(mixed)], strict=True):
                    ax.scatter(
                        x[s],
                        acc[s],
                        s=52,
                        color=ORANGE,
                        marker=marker,
                        edgecolor=SURFACE,
                        lw=1.5,
                        zorder=3,
                        label=s.replace("mixed:", "mixed precision, "),
                    )
                ax.set_xscale("log")
                ax.set_xlabel(xlabel)
            row_axes[0].set_ylabel(f"{condition}\nheld-out-day accuracy (%)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3)
        fig.suptitle("Accuracy against cost; points on the blue line are labelled with their bit width")
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(OUT / "fig_cost_accuracy.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------- summary.md


def signed(value: float) -> str:
    """'+0.12'; a value that rounds to zero prints as +0.00, never -0.00."""
    return f"{round(value, 2) + 0.0:+.2f}"


def interval(row, prefix: str = "") -> str:
    """'+0.12 [-0.05, +0.30]': a mean difference and its 95% participant bootstrap interval."""
    return (
        f"{signed(row[prefix + 'diff_pp'])} [{signed(row[prefix + 'ci_low_pp'])}, {signed(row[prefix + 'ci_high_pp'])}]"
    )


def listing(names: list[str], kind: str) -> str:
    return ", ".join(plain(kind, n) for n in names) or "none"


def agreement(found: list[str], planned: list[str], kind: str) -> str:
    extra, spared = [n for n in found if n not in planned], [n for n in planned if n not in found]
    parts = [f"the rule also keeps {listing(extra, kind)}"] if extra else []
    parts += [f"the plan also keeps {listing(spared, kind)}, which the rule would round"] if spared else []
    return "; ".join(parts) or "the same set"


def number(value, spec: str) -> str:
    return "n/a" if pd.isna(value) else format(value, spec)


def intro_lines(counts: dict, plan: dict) -> list[str]:
    q = plan["qat"]
    qat = (
        f"Quantization-aware training (QAT) fine-tuned {len(q['conditions'])} architectures "
        f"({', '.join(q['conditions'])}) for {q['epochs']} more epochs with the rounding in the loop, next to a "
        f"matched control fine-tuned for the same epochs without rounding ({counts['qat']} runs)."
        if counts["qat"]
        else "Quantization-aware training (QAT) is left out of this run (--ptq-only)."
    )
    return [
        "# Bit-width study (supplementary, post-test; generated by scripts/analyze_quant_study.py)",
        "",
        "How many bits does the decoder need, whatever chip runs it? Every archived final model "
        f"({counts['ptq']}: {counts['conditions']} conditions x {counts['participants']} participants x "
        f"{counts['seeds']} seeds) was rounded to a grid of weight and activation bit widths without retraining "
        f"(post-training quantization, PTQ). {qat}",
        "",
        "## How to read the numbers",
        "",
        "- W/A means weight bits / activation bits, and float is 32-bit floating point: 4/8 is 4-bit weights with "
        "8-bit activations. Weights are rounded per output channel; activations per tensor, over the range seen on "
        "the model's own session-1 training trials.",
        "- Layers in order: input EEG, conv_t (temporal convolution), conv_s (spatial convolution), ELU, pooling, "
        "conv_d (depthwise convolution), conv_p (pointwise convolution), ELU, pooling, fc (classifier), logits. "
        "BatchNorm is folded into the convolutions, as in the exported graphs.",
        "- Accuracy is on the artifact-free trials of session 2, the held-out day. Seeds are averaged within each "
        "participant (in pooled rows, the conditions too); then the nine participants are averaged.",
        "- Differences are in percentage points (pp). Brackets hold the 95% participant bootstrap interval of the "
        "frozen plan (10,000 resamples, seed 690). 'Worst' is the participant who loses most; 'n > 2' counts the "
        "participants who lose more than 2 points.",
        "",
        "## Boundaries",
        "",
        "- Bit widths below 8 are simulated: weights and activations are rounded to integer grids and the network "
        "is evaluated in floating point (mibci.fakequant). No low-bit kernel was run or timed, so nothing here is "
        "about speed.",
        "- All session-2 numbers are exploratory: the study was planned after the main study's test results were "
        "known.",
        "- The precision rule (which tensors keep more bits) is read from session-1 sensitivity only. Session 2 "
        "only shows whether it carries over to the held-out day.",
        "- Costs are counted from the network's shape, not measured. The energy figure counts arithmetic only, "
        "not memory traffic, and is an order-of-magnitude estimate.",
    ]


def simulator_lines(check: dict) -> list[str]:
    return [
        "",
        "## Does the simulation reproduce ONNX Runtime INT8?",
        "",
        "The simulated 8/8 network uses the grids of the archived ONNX Runtime INT8 graphs (per-channel 8-bit "
        "weights, min-max activation ranges from the same session-1 trials), so the two should agree.",
        "",
        "- Predictions agree with the archived INT8 predictions on "
        f"{check['agreement_with_archived_int8_pct_mean']:.2f}% of session-2 trials on average over the "
        f"{check['models']} models (lowest model "
        f"{check['agreement_with_archived_int8_pct_min']:.2f}%), and with the archived FP32 predictions on "
        f"{check['agreement_with_archived_fp32_pct_mean']:.2f}%.",
        f"- Accuracy differs from ONNX Runtime INT8 by {check['abs_acc_diff_sim_w8a8_vs_ort_int8_pp_mean']:.2f} pp per "
        f"model on average (largest {check['abs_acc_diff_sim_w8a8_vs_ort_int8_pp_max']:.2f} pp). Over participants, "
        f"simulated minus ONNX Runtime INT8 is {interval(check['sim_w8a8_minus_ort_int8'])} pp.",
        "- Without any rounding, the folded network matches the archived FP32 accuracy to "
        f"{check['abs_acc_diff_float_vs_archived_fp32_pp_mean']:.2f} pp per model on average.",
    ]


def ptq_lines(table: pd.DataFrame, plan: dict, narrow: dict) -> list[str]:
    pooled = table[table.condition == PTQ_POOLED].set_index("setting")
    activations = plan["activation_bits"]
    lines = [
        "",
        "## Accuracy against bit width, pooled over all conditions (minus float/float, pp)",
        "",
        f"Float/float accuracy is {pooled.loc[FLOAT, 'acc_mean']:.2f}%. Rows are weight bits, columns activation bits; "
        "per-condition values are in ptq_grid.csv and fig_ptq_heatmap.png.",
        "",
        "| W \\ A | " + " | ".join(activations) + " |",
        "| --- |" + " ---: |" * len(activations),
    ]
    for w in plan["weight_bits"]:
        cells = " | ".join(signed(pooled.loc[f"{w}/{a}", "vs_float_diff_pp"]) for a in activations)
        lines.append(f"| {w} | {cells} |")
    names = list(families(plan))
    lines += [
        "",
        "## Narrowest width at which no participant loses more than 2 points",
        "",
        "Going from wide to narrow along each line of the grid: the last setting before the first one at which some "
        "participant loses more than 2 points against float/float (session 2, exploratory). 'none' means the widest "
        "setting of the line already fails.",
        "",
        "| condition | " + " | ".join(names) + " |",
        "| --- |" + " --- |" * len(names),
    ]
    for condition, found in narrow.items():
        lines.append(f"| {condition} | " + " | ".join(found[n] or "none" for n in names) + " |")
    lines += [
        "",
        "## Key settings, pooled over all conditions",
        "",
        "| setting | accuracy % | minus float/float (pp) [95% interval] | worst (pp) | n > 2 |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for s in [*KEY_SETTINGS, *plan["mixed"]]:
        r = pooled.loc[s]
        lines.append(
            f"| {s} | {r.acc_mean:.2f} | {interval(r, 'vs_float_')} | {signed(r.vs_float_worst_participant_pp)} | "
            f"{r.vs_float_n_losing_over_2pp} |"
        )
    return lines


def sensitivity_lines(table: pd.DataFrame, rule: dict) -> list[str]:
    pooled = table[table.condition == PTQ_POOLED]
    s1 = pooled[pooled.session == "session1"].set_index("tensor")
    s2 = pooled[pooled.session == "session2"].set_index("tensor")
    rho = rule["spearman_session1_vs_session2"]
    lines = [
        "",
        "## Which tensors need more bits (one tensor rounded, everything else in floating point)",
        "",
        "Session 1 means the model's own artifact-free session-1 trials, the trials it was trained on, so accuracy "
        "there is high. It needs no test data, so the precision rule is read from it. Session 2 shows whether the "
        "ranking carries over. Pooled over all conditions; per condition in sensitivity.csv.",
        "",
        "| tensor rounded alone | session 1: change (pp) [95%] | session 2: change (pp) [95%] |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {s1.loc[t, 'description']} | {interval(s1.loc[t])} | {interval(s2.loc[t])} |" for t in s1.index]
    acts, planned_acts = rule["activations_need_more_than_4_bits"], rule["planned_activations_at_8_bits"]
    weights_2, planned_weights = rule["weights_need_more_than_2_bits"], rule["planned_weights_at_4_bits"]
    weights_4 = rule["weights_need_more_than_4_bits"]
    lines += [
        "",
        f"Rank correlation (Spearman) of the session-1 and session-2 changes over these {rho['n']} cases: "
        f"{rho['rho']:+.2f} (p = {rho['p']:.2g}).",
        "",
        f"Rule, from session 1 only: a tensor needs more bits when rounding it alone costs more than "
        f"{rule['threshold_pp']:g} points of training-trial accuracy, on average over participants.",
        "",
        f"- Activations that need more than 4 bits, so the rule keeps them at 8: {listing(acts, 'activations')}. "
        f"The planned early-8 settings keep {listing(planned_acts, 'activations')} at 8 bits: "
        f"{agreement(acts, planned_acts, 'activations')}.",
        f"- Weights that need more than 4 bits, so the rule keeps them at 8: {listing(weights_4, 'weights')}"
        + ("; every planned mixed setting rounds all weights to 4 bits or fewer." if weights_4 else "."),
        f"- Weights that need more than 2 bits, so the rule keeps them at 4: {listing(weights_2, 'weights')}. "
        f"The planned spatial-4 settings keep {listing(planned_weights, 'weights')} at 4 bits: "
        f"{agreement(weights_2, planned_weights, 'weights')}.",
    ]
    return lines


def mixed_lines(mixed: pd.DataFrame, plan: dict) -> list[str]:
    lines = ["", "## Mixed precision", ""]
    lines += [
        f"- {key}: weights {bits_label(spec['weights'])}; activations {bits_label(spec['activations'])}"
        for key, spec in plan["mixed"].items()
    ]
    lines += [
        "",
        "| condition | setting | accuracy % | minus float/float (pp) [95%] | n > 2 | minus 8/8 (pp) [95%] "
        "| minus 4/4 (pp) [95%] |",
        "| --- | --- | ---: | --- | ---: | --- | --- |",
    ]
    for condition in [PTQ_POOLED, *plan["qat"]["conditions"]]:
        for setting in plan["mixed"]:
            r = mixed[(mixed.condition == condition) & (mixed.setting == setting)].set_index("reference")
            lines.append(
                f"| {condition} | {setting} | {r.loc[FLOAT, 'acc_mean']:.2f} | {interval(r.loc[FLOAT])} | "
                f"{r.loc[FLOAT, 'n_losing_over_2pp']} | {interval(r.loc[W8A8])} | {interval(r.loc[W4A4])} |"
            )
    return lines


def qat_lines(qat: pd.DataFrame | None, plan: dict) -> list[str]:
    if qat is None:
        return ["", "## Quantization-aware fine-tuning", "", "Left out of this run (--ptq-only)."]
    q = plan["qat"]
    lines = [
        "",
        f"## Quantization-aware fine-tuning ({len(q['conditions'])} architectures, {q['epochs']} epochs, "
        f"Adam, learning rate {q['lr']:g})",
        "",
        "Each variant starts from the archived model with its rounding switched on and trains for "
        f"{q['epochs']} more epochs on the session-1 training trials. {CONTROL} trains the same epochs without "
        "rounding; it is the fair reference, because extra training alone changes accuracy. 'Before' is the same "
        f"rounding without fine-tuning (PTQ); for {CONTROL} it is the archived model itself.",
        "",
        f"| condition | variant | before % | after % | after minus {CONTROL} (pp) [95%] | worst (pp) | n > 2 "
        "| after minus before (pp) [95%] |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for _, r in qat.iterrows():
        lines.append(
            f"| {r.condition} | {r.variant} | {r.acc_before:.2f} | {r.acc_after:.2f} | {interval(r, 'vs_fp32ft_')} | "
            f"{signed(r.vs_fp32ft_worst_participant_pp)} | {r.vs_fp32ft_n_losing_over_2pp} | "
            f"{interval(r, 'vs_before_')} |"
        )
    return lines


def cost_lines(cost: pd.DataFrame, plan: dict) -> list[str]:
    first = architecture_rows(cost, SHOWN[0]).loc[W8A8]
    lines = [
        "",
        "## Cost of one inference (one 2-s window, batch 1; counted, not measured)",
        "",
        "- Weight storage: weights x weight bits / 8, plus every bias at 32 bits, BatchNorm folded as deployed. "
        "Not included: one 32-bit scale per output channel of each rounded layer "
        f"(weight_scale_bytes in costs.csv, {first.weight_scale_bytes:,.0f} bytes for {SHOWN[0]}) and one scale and "
        "zero point per activation tensor.",
        "- Peak activation memory: the largest input + output pair of one layer (as in costs.py), each tensor at "
        "its own width.",
        "- Bit-operations: MACs x weight bits x activation bits, floating point counted as 32 bits.",
        f"- Arithmetic energy, Horowitz (ISSCC 2014, 45 nm): an integer multiply costs {INT8_MULTIPLY_PJ:g} pJ at 8 "
        f"bits, taken to scale with the product of the two widths ({INT8_MULTIPLY_PJ:g} x w/8 x a/8 pJ, which gives "
        f"3.2 pJ at 32 bits against the reported 3.1); the accumulate is a 32-bit integer add, {INT32_ADD_PJ:g} pJ; a "
        f"layer with floating-point weights or activations costs a float32 multiply and add, "
        f"{FLOAT32_MULTIPLY_PJ:g} + {FLOAT32_ADD_PJ:g} pJ per MAC. Only MACs are counted. Memory traffic (reading "
        "weights and activations), which on real chips often costs more than the arithmetic, is not counted: "
        "treat the energy as an order-of-magnitude estimate, not a measurement.",
        "- Layer-swapped: the same function with the spatial filters first (results/supplementary/int8_study); its "
        + (
            "MACs are read from that study's costs.csv. Its energy is shown only where all weights share one width "
            "and all activations another."
            if cost.spatial_first_macs.notna().any()
            else f"costs are not shown, because {display(INT8_COSTS)} was not found."
        ),
        "",
        "| condition | setting | weight bytes | peak activation bytes | MACs | bit-operations (millions) "
        "| energy (µJ) | layer-swapped MACs | layer-swapped energy (µJ) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in SHOWN:
        rows = architecture_rows(cost, condition)
        for s in [*COST_SETTINGS, *plan["mixed"]]:
            r = rows.loc[s]
            lines.append(
                f"| {condition} | {s} | {r.weight_bytes:,.0f} | {r.peak_activation_bytes:,.0f} | {r.macs:,} | "
                f"{r.bit_operations / 1e6:,.1f} | {r.mac_energy_uj:.3g} | {number(r.spatial_first_macs, ',.0f')} | "
                f"{number(r.spatial_first_mac_energy_uj, '.3g')} |"
            )
    return lines


def key_lines(counts: dict, check: dict, table: pd.DataFrame, narrow: dict, rule: dict, qat, cost) -> list[str]:
    """The headline numbers of each section, in the order of the sections below."""
    pooled = table[table.condition == PTQ_POOLED].set_index("setting")
    same, with8, _ = narrow[PTQ_POOLED].values()
    lines = [
        "",
        "## Key numbers",
        "",
        "- The simulated 8/8 network agrees with the archived ONNX Runtime INT8 predictions on "
        f"{check['agreement_with_archived_int8_pct_mean']:.2f}% of session-2 trials on average (lowest model "
        f"{check['agreement_with_archived_int8_pct_min']:.2f}%).",
        f"- Pooled over all conditions, against float/float: 8/8 {interval(pooled.loc[W8A8], 'vs_float_')} pp, "
        f"4/8 {interval(pooled.loc['4/8'], 'vs_float_')} pp, 4/4 {interval(pooled.loc[W4A4], 'vs_float_')} pp; at "
        f"4/4, {pooled.loc[W4A4, 'vs_float_n_losing_over_2pp']} of {counts['participants']} participants lose more "
        "than 2 points.",
        "- Narrowest setting at which no participant loses more than 2 points, pooled: "
        f"{same or 'none'} with equal widths, {with8 or 'none'} with 8-bit activations.",
        "- Session-1 rule: activations kept at 8 bits: "
        f"{listing(rule['activations_need_more_than_4_bits'], 'activations')}; weights kept above 2 bits: "
        f"{listing(rule['weights_need_more_than_2_bits'], 'weights')}. Rank correlation of the session-1 and "
        f"session-2 sensitivities: {rule['spearman_session1_vs_session2']['rho']:+.2f}.",
    ]
    if qat is not None:
        q = qat[qat.condition == QAT_POOLED].set_index("variant")
        lines.append(
            f"- QAT, pooled over the QAT conditions, after minus the matched {CONTROL} control (pp): "
            + ", ".join(f"{v} {signed(q.loc[v, 'vs_fp32ft_diff_pp'])}" for v in q.index if v != CONTROL)
            + "."
        )
    rows = architecture_rows(cost, SHOWN[1])
    full, eight = rows.loc[FLOAT], rows.loc[W8A8]
    swapped = "" if pd.isna(eight.spatial_first_mac_energy_uj) else f", {eight.spatial_first_mac_energy_uj:.3g} µJ"
    lines.append(
        f"- {SHOWN[1]} (counted, not measured): weights take {full.weight_bytes:,.0f} bytes in float and "
        f"{eight.weight_bytes:,.0f} at 8 bits; arithmetic per window is about {full.mac_energy_uj:.3g} µJ in float "
        f"and {eight.mac_energy_uj:.3g} µJ at 8/8" + (f"{swapped} with the layer swap" if swapped else "") + "."
    )
    return lines


def files_lines(qat_included: bool) -> list[str]:
    return [
        "",
        "## Files",
        "",
        "ptq_grid.csv (every setting and condition), sensitivity.csv (every condition), mixed.csv (every condition), "
        + ("qat.csv, " if qat_included else "")
        + "costs.csv (every architecture and setting), fig_ptq_heatmap.png, fig_sensitivity.png, "
        "fig_cost_accuracy.png; analysis.json holds the numbers quoted here.",
    ]


# ---------------------------------------------------------------- main


def records_of(df: pd.DataFrame) -> list[dict]:
    """Rows for analysis.json, with NaN as null."""
    return json.loads(df.to_json(orient="records"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--ptq-only", action="store_true", help="leave out QAT, e.g. to look at PTQ while QAT is still running"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = load_plan()
    boot = frozen_config()["uncertainty"]
    settings = settings_of(plan)
    ptq = read_records(plan, "ptq")
    grid, sens, sim = ptq_frames(ptq, settings)
    acc, acc_all = ptq_accuracy(grid)
    table = ptq_grid(acc, acc_all, settings, boot)
    sensitivity = sensitivity_table(sens, boot)
    rule = precision_rule(sensitivity, plan)
    mixed = mixed_table(acc, plan, boot)
    qat_records = [] if args.ptq_only else read_records(plan, "qat")
    qat = qat_table(qat_frame(qat_records), plan, boot) if qat_records else None
    cost = cost_table(grid.condition.unique(), settings)
    check = simulator_check(sim, boot)
    narrow = narrowest_safe(table, plan)

    table.to_csv(OUT / "ptq_grid.csv", index=False)
    sensitivity.to_csv(OUT / "sensitivity.csv", index=False)
    mixed.to_csv(OUT / "mixed.csv", index=False)
    if qat is not None:
        qat.to_csv(OUT / "qat.csv", index=False)
    cost.to_csv(OUT / "costs.csv", index=False)
    figure_heatmap(table, plan)
    figure_sensitivity(sensitivity)
    figure_cost_accuracy(table, cost, plan, settings)

    counts = {
        "ptq": len(ptq),
        "qat": len(qat_records),
        "conditions": grid.condition.nunique(),
        "participants": grid.subject.nunique(),
        "seeds": grid.seed.nunique(),
    }
    lines = [
        *intro_lines(counts, plan),
        *key_lines(counts, check, table, narrow, rule, qat, cost),
        *simulator_lines(check),
        *ptq_lines(table, plan, narrow),
        *sensitivity_lines(sensitivity, rule),
        *mixed_lines(mixed, plan),
        *qat_lines(qat, plan),
        *cost_lines(cost, plan),
        *files_lines(qat is not None),
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    shown_mixed = mixed[mixed.condition.isin([PTQ_POOLED, *plan["qat"]["conditions"]])]
    shown_costs = pd.concat(
        architecture_rows(cost, c).loc[[*COST_SETTINGS, *plan["mixed"]]].reset_index() for c in SHOWN
    )
    write_json(
        OUT / "analysis.json",
        {
            "status": "supplementary_post_test",
            "records": counts,
            "bootstrap": boot,
            "material_loss_pp": MATERIAL_PP,
            "simulator_check": check,
            "ptq_pooled": records_of(table[table.condition == PTQ_POOLED]),
            "narrowest_safe": narrow,
            "sensitivity_pooled": records_of(sensitivity[sensitivity.condition == PTQ_POOLED]),
            "precision_rule": rule,
            "mixed": records_of(shown_mixed),
            "qat": None if qat is None else records_of(qat),
            "costs": records_of(shown_costs),
            "energy_pj": {
                "int8_multiply": INT8_MULTIPLY_PJ,
                "int32_add": INT32_ADD_PJ,
                "float32_multiply": FLOAT32_MULTIPLY_PJ,
                "float32_add": FLOAT32_ADD_PJ,
                "source": "Horowitz, ISSCC 2014, 45 nm; arithmetic only, no memory traffic; order of magnitude",
            },
            "note": "Exploratory: planned after the main study's test results were known. Bit widths below 8 are "
            "simulated (integer grids in floating point); no low-bit kernel was timed. The precision rule is read "
            "from session-1 sensitivity only; session 2 is reported to show whether it carries over.",
        },
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
