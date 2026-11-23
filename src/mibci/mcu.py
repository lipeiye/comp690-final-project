"""EEGNet on Cortex-M microcontrollers: export, TFLite-style int8 quantization and a bit-exact reference.

scripts/mcu_study.py runs trained EEGNets with Arm's CMSIS-NN kernels on emulated Cortex-M cores. This
module turns a trained network into the tensors those kernels take, writes them as C headers, and
reproduces the kernels' integer arithmetic in NumPy, so that every emulated output can be checked bit
for bit. Layouts follow TensorFlow Lite (NHWC: height, width, channels), which CMSIS-NN implements:

  original order  conv_t  CONV_2D    (C, T, 1)     -> (C, T+1, F1)   kernel 1x64, zero padding 32
                  conv_s  DEPTHWISE  (C, T+1, F1)  -> (1, T+1, F1*D) kernel Cx1, channel multiplier D
  swapped order   conv_s  CONV_2D    (1, T, C)     -> (1, T, F1*D)   1x1, the electrodes are channels
                  conv_t  DEPTHWISE  (1, T, F1*D)  -> (1, T+1, F1*D) kernel 1x64, multiplier 1
  both            elu1 (256-entry table), pool1 AVERAGE_POOL 1x4, conv_d DEPTHWISE 1x16 padding 8,
                  conv_p CONV_2D 1x1 -> F2, elu2, pool2 1x8, fc FULLY_CONNECTED -> classes

BatchNorm is folded as in mibci.fakequant.Folded; the swap is the algebra of
mibci.variants.spatial_first. Quantization follows the TFLite int8 specification: per-tensor asymmetric
activations from min-max calibration, per-channel symmetric convolution weights, per-tensor fully
connected weights, int32 biases and fixed-point output multipliers; average pooling keeps its input's
scale, and ELU is a lookup table built from the scales on each side. Requantization is CMSIS-NN's
default (two rounding steps, no inline assembly, CMSIS_NN_USE_SINGLE_ROUNDING undefined).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from mibci import fakequant

ORDERS = ("original", "swapped")
# tensors whose min-max range sets a scale: network input, after each weighted layer and each ELU
TAPS = ("input", "l1", "l2", "elu1", "conv_d", "conv_p", "elu2", "logits")
LAYERS = ("l1", "l2", "elu1", "pool1", "conv_d", "conv_p", "elu2", "pool2", "fc")


def export(model, order: str) -> dict:
    """Float tensors of a trained EEGNet (BatchNorm folded) for one layer order, in TFLite layouts."""
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}")
    f = fakequant.Folded(model)

    def arr(t):
        return t.detach().double().numpy()

    wt, bt = arr(f.conv_t.weight)[:, 0, 0, :], arr(f.conv_t.bias)  # (F1, K), (F1,)
    ws, bs = arr(f.conv_s.weight)[:, 0, :, 0], arr(f.conv_s.bias)  # (F1*D, C), (F1*D,)
    wd = arr(f.conv_d.weight)[:, 0, 0, :]  # (F1*D, 16); no bias
    wp, bp = arr(f.conv_p.weight)[:, :, 0, 0], arr(f.conv_p.bias)  # (F2, F1*D)
    wf, bf = arr(f.fc.weight), arr(f.fc.bias)  # (classes, F2 * W2), PyTorch flattens [f][t]
    f1, k = wt.shape
    f1d, n_chans = ws.shape
    depth, f2 = f1d // f1, wp.shape[0]
    w2 = wf.shape[1] // f2
    fc_nhwc = wf.reshape(len(bf), f2, w2).transpose(0, 2, 1).reshape(len(bf), -1)  # NHWC flattens [t][f]
    if order == "original":
        first = [
            {"name": "l1", "op": "conv", "w": wt[:, None, :, None], "b": bt, "pad": (0, k // 2)},
            {"name": "l2", "op": "dw", "w": ws.T[:, None, :], "b": bs, "pad": (0, 0), "mult": depth},
        ]
    else:
        temporal = np.repeat(wt, depth, axis=0)  # output o uses temporal filter o // D
        first = [
            {"name": "l1", "op": "conv", "w": ws[:, None, None, :], "b": np.zeros(f1d), "pad": (0, 0)},
            {"name": "l2", "op": "dw", "w": temporal.T[None], "b": np.repeat(bt, depth) * ws.sum(1) + bs,
             "pad": (0, k // 2), "mult": 1},
        ]  # fmt: skip
    rest = [
        {"name": "elu1", "op": "elu"},
        {"name": "pool1", "op": "pool", "k": int(f.pool1.kernel_size[1])},
        {"name": "conv_d", "op": "dw", "w": wd.T[None], "b": np.zeros(f1d), "pad": (0, wd.shape[1] // 2), "mult": 1},
        {"name": "conv_p", "op": "conv", "w": wp[:, None, None, :], "b": bp, "pad": (0, 0)},
        {"name": "elu2", "op": "elu"},
        {"name": "pool2", "op": "pool", "k": int(f.pool2.kernel_size[1])},
        {"name": "fc", "op": "fc", "w": fc_nhwc, "b": bf},
    ]
    return {"order": order, "n_chans": n_chans, "f1": f1, "depth": depth, "f2": f2, "layers": first + rest}


def to_nhwc(X: np.ndarray, order: str) -> np.ndarray:
    """(trials, electrodes, samples) -> NHWC network input: (C, T, 1) originally, (1, T, C) swapped."""
    return X[..., None] if order == "original" else X.transpose(0, 2, 1)[:, None]


def _windows(x: np.ndarray, kh: int, kw: int, pad: tuple[int, int]) -> np.ndarray:
    x = np.pad(x, ((0, 0), (pad[0], pad[0]), (pad[1], pad[1]), (0, 0)))
    return sliding_window_view(x, (kh, kw), axis=(1, 2))  # (N, Ho, Wo, channels, kh, kw)


def _conv(x: np.ndarray, layer: dict) -> np.ndarray:
    w = layer["w"]  # (Cout, KH, KW, Cin)
    win = _windows(x, w.shape[1], w.shape[2], layer["pad"])
    return np.tensordot(win, w.transpose(3, 1, 2, 0), axes=([3, 4, 5], [0, 1, 2]))


def _dw(x: np.ndarray, layer: dict) -> np.ndarray:
    w = layer["w"]  # (KH, KW, Cout)
    win = _windows(np.repeat(x, layer["mult"], axis=-1), w.shape[0], w.shape[1], layer["pad"])
    return np.einsum("nhwoij,ijo->nhwo", win, w, optimize=True)


def _pool(x: np.ndarray, k: int) -> np.ndarray:
    n, h, w, c = x.shape
    return x[:, :, : w // k * k].reshape(n, h, w // k, k, c).sum(3)


def float_forward(net: dict, X: np.ndarray, taps: dict | None = None, batch: int = 32) -> np.ndarray:
    """Float logits of the exported network on standardized trials; `taps` collects tensors by name.
    Without taps the trials go through in batches, which bounds the memory of the unfolded windows."""
    if taps is None and len(X) > batch:
        return np.concatenate([float_forward(net, X[i : i + batch], batch=batch) for i in range(0, len(X), batch)])
    x = to_nhwc(X.astype(np.float64), net["order"])
    if taps is not None:
        taps["input"] = x
    for layer in net["layers"]:
        op = layer["op"]
        if op == "conv":
            x = _conv(x, layer) + layer["b"]
        elif op == "dw":
            x = _dw(x, layer) + layer["b"]
        elif op == "elu":
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
        elif op == "pool":
            x = _pool(x, layer["k"]) / layer["k"]
        else:
            x = x.reshape(len(x), -1) @ layer["w"].T + layer["b"]
        if taps is not None and layer["name"] in TAPS:
            taps[layer["name"]] = x
    if taps is not None:
        taps["logits"] = x
    return x


def calibrate(net: dict, X: np.ndarray, batch: int = 64) -> dict[str, list[float]]:
    """Min and max of every tapped tensor over the calibration trials (TFLite min-max calibration)."""
    ranges = {name: [np.inf, -np.inf] for name in TAPS}
    for i in range(0, len(X), batch):
        taps: dict = {}
        float_forward(net, X[i : i + batch], taps)
        for name in TAPS:
            ranges[name] = [
                min(ranges[name][0], float(taps[name].min())),
                max(ranges[name][1], float(taps[name].max())),
            ]
    return ranges


def round_half_away(x):
    """std::round, which TFLite uses when it quantizes: halves go away from zero."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def activation_params(low: float, high: float) -> tuple[float, int]:
    """Scale and zero point of an asymmetric int8 tensor covering [low, high] and zero (TFLite)."""
    low, high = min(low, 0.0), max(high, 0.0)
    scale = (high - low) / 255.0 if high > low else 1.0
    zero = int(np.clip(round_half_away(-128 - low / scale), -128, 127))
    return scale, zero


def quantize_multiplier(m: float) -> tuple[int, int]:
    """TFLite QuantizeMultiplier: m = q * 2^shift with q a Q31 number in [0.5, 1); shift > 0 is a left shift."""
    if m == 0:
        return 0, 0
    q, shift = math.frexp(m)
    fixed = round(q * (1 << 31))
    if fixed == 1 << 31:
        fixed //= 2
        shift += 1
    if shift < -31:
        return 0, 0
    return int(fixed), int(shift)


def _weights(w: np.ndarray, per_channel_axis: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int8 weights (levels -127..127) with one scale per channel, or one per tensor."""
    if per_channel_axis is None:
        scale = np.array([max(np.abs(w).max(), 1e-12) / 127.0])
        return np.clip(round_half_away(w / scale[0]), -127, 127).astype(np.int8), scale
    axes = tuple(i for i in range(w.ndim) if i != per_channel_axis)
    scale = np.maximum(np.abs(w).max(axis=axes), 1e-12) / 127.0
    shape = [1] * w.ndim
    shape[per_channel_axis] = -1
    return np.clip(round_half_away(w / scale.reshape(shape)), -127, 127).astype(np.int8), scale


def _elu_table(s_in: float, z_in: int, s_out: float, z_out: int) -> np.ndarray:
    x = s_in * (np.arange(-128, 128) - z_in)
    y = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
    return np.clip(round_half_away(y / s_out) + z_out, -128, 127).astype(np.int8)


def quantize(net: dict, ranges: dict[str, list[float]]) -> dict:
    """The int8 network: per layer the tensors and constants CMSIS-NN takes."""
    params = {name: activation_params(*ranges[name]) for name in TAPS}
    scale, zero = params["input"]
    layers = []
    for layer in net["layers"]:
        op, name = layer["op"], layer["name"]
        q = {"name": name, "op": op, "in_scale": scale, "in_zero": zero}
        if op == "pool":
            q["k"] = layer["k"]  # output keeps the input's scale and zero point
        elif op == "elu":
            out_scale, out_zero = params[name]
            q["table"] = _elu_table(scale, zero, out_scale, out_zero)
            scale, zero = out_scale, out_zero
        else:
            axis = {"conv": 0, "dw": 2, "fc": None}[op]
            out_scale, out_zero = params["logits" if op == "fc" else name]
            wq, w_scale = _weights(layer["w"], axis)
            effective = scale * w_scale
            q |= {"w": wq, "b": round_half_away(layer["b"] / effective).astype(np.int64), "pad": layer.get("pad"),
                  "mult": layer.get("mult"), "out_scale": out_scale, "out_zero": out_zero}  # fmt: skip
            fixed = [quantize_multiplier(m) for m in effective / out_scale]
            q["multiplier"] = np.array([m for m, _ in fixed], dtype=np.int64)
            q["shift"] = np.array([s for _, s in fixed], dtype=np.int64)
            if np.abs(q["b"]).max() >= 2**31:
                raise ValueError(f"{name}: bias does not fit in int32")
            scale, zero = out_scale, out_zero
        layers.append(q)
    return {**{k: v for k, v in net.items() if k != "layers"}, "layers": layers, "params": params}


def quantize_input(X: np.ndarray, qnet: dict) -> np.ndarray:
    """Standardized trials -> int8 NHWC input of the quantized network."""
    scale, zero = qnet["params"]["input"]
    q = np.clip(round_half_away(to_nhwc(X.astype(np.float64), qnet["order"]) / scale) + zero, -128, 127)
    return q.astype(np.int8)


def _wrap32(x: np.ndarray) -> np.ndarray:
    return ((x + 2**31) % 2**32) - 2**31


def requantize(acc: np.ndarray, multiplier: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """arm_nn_requantize: doubling high multiply with rounding, then a rounding right shift."""
    left, right = np.maximum(shift, 0), np.maximum(-shift, 0)
    value = _wrap32(acc * (np.int64(1) << left))  # int32 product in C
    high = _wrap32((value * multiplier + (1 << 30)) >> 31)
    mask = (np.int64(1) << right) - 1
    result = high >> right
    threshold = (mask >> 1) + (result < 0)
    return result + ((high & mask) > threshold)


def int_forward(qnet: dict, xq: np.ndarray, batch: int = 32) -> np.ndarray:
    """int8 logits, computed as CMSIS-NN computes them (NHWC int8 input from quantize_input)."""
    if len(xq) > batch:
        return np.concatenate([int_forward(qnet, xq[i : i + batch], batch) for i in range(0, len(xq), batch)])
    x = xq.astype(np.int64)
    for q in qnet["layers"]:
        op = q["op"]
        if op == "elu":
            x = q["table"][x + 128].astype(np.int64)
            continue
        if op == "pool":
            total = _pool(x, q["k"])
            half = q["k"] // 2
            x = np.clip(np.trunc(np.where(total > 0, total + half, total - half) / q["k"]), -128, 127).astype(np.int64)
            continue
        shifted = (x - q["in_zero"]).astype(np.float64)  # padding below is real zero, i.e. the zero point
        w = q["w"].astype(np.float64)
        if op == "conv":
            acc = _conv(shifted, {"w": w, "pad": q["pad"]})
        elif op == "dw":
            acc = _dw(shifted, {"w": w, "pad": q["pad"], "mult": q["mult"]})
        else:
            acc = shifted.reshape(len(shifted), -1) @ w.T
        acc = np.rint(acc).astype(np.int64) + q["b"]  # every sum is an exact integer in float64
        x = np.clip(requantize(acc, q["multiplier"], q["shift"]) + q["out_zero"], -128, 127)
    return x.astype(np.int8)


def dequantize_logits(logits: np.ndarray, qnet: dict) -> np.ndarray:
    scale, zero = qnet["params"]["logits"]
    return (logits.astype(np.float64) - zero) * scale


def shapes(net: dict, n_times: int) -> list[dict]:
    """Per layer: NHWC input and output shapes, multiply-accumulates and stored values."""
    x = np.zeros((1, net["n_chans"], n_times))
    rows = []
    shape = to_nhwc(x, net["order"]).shape[1:]
    for layer in net["layers"]:
        op = layer["op"]
        h, w, c = shape
        if op == "conv":
            cout, kh, kw, cin = layer["w"].shape
            out = (h + 2 * layer["pad"][0] - kh + 1, w + 2 * layer["pad"][1] - kw + 1, cout)
            macs = int(np.prod(out)) * kh * kw * cin
        elif op == "dw":
            kh, kw, cout = layer["w"].shape
            out = (h + 2 * layer["pad"][0] - kh + 1, w + 2 * layer["pad"][1] - kw + 1, cout)
            macs = int(np.prod(out)) * kh * kw
        elif op == "pool":
            out, macs = (h, w // layer["k"], c), 0
        elif op == "elu":
            out, macs = shape, 0
        else:
            out, macs = (1, 1, layer["w"].shape[0]), int(layer["w"].size)
        values = int(layer["w"].size + layer["b"].size) if "w" in layer else 0
        rows.append({"name": layer["name"], "op": op, "in": list(shape), "out": list(out), "macs": macs,
                     "weights": values})  # fmt: skip
        shape = out
    return rows


def buffers(rows: list[dict]) -> tuple[int, int]:
    """Elements of the two ping-pong activation buffers: the input in A, each layer writes the other
    buffer, and ELU works in place."""
    sizes = [int(np.prod(rows[0]["in"])), 0]
    current = 0
    for row in rows:
        if row["op"] == "elu":
            continue
        current = 1 - current
        sizes[current] = max(sizes[current], int(np.prod(row["out"])))
    return sizes[0], sizes[1]


def _c_value(ctype: str, v) -> str:
    return str(int(v)) if ctype != "float" else f"{float(np.float32(v))!r}f"


def _c_array(ctype: str, name: str, values, per_line: int = 24) -> str:
    flat = [_c_value(ctype, v) for v in np.ravel(values)]
    lines = [", ".join(flat[i : i + per_line]) for i in range(0, len(flat), per_line)]
    return f"static const {ctype} {name}[{len(flat)}] = {{\n    " + ",\n    ".join(lines) + "\n};\n"


def _dims_header(net: dict, rows: list[dict], n_classes: int, out_bytes: int) -> str:
    by = {r["name"]: r for r in rows}
    a, b = buffers(rows)
    temporal = net["layers"][0 if net["order"] == "original" else 1]["w"]  # conv (F1, 1, K, 1) or dw (1, K, F1*D)
    values = {
        "MODEL_SWAPPED": int(net["order"] == "swapped"),
        "N_CHANS": net["n_chans"],
        "N_TIMES": rows[0]["in"][1],
        "N_CLASSES": n_classes,
        "OUT_BYTES": out_bytes,
        "F1": net["f1"],
        "DEPTH": net["depth"],
        "F1D": net["f1"] * net["depth"],
        "F2": net["f2"],
        "W0": by["l2"]["out"][1],
        "W1": by["pool1"]["out"][1],
        "WD": by["conv_d"]["out"][1],
        "W2": by["pool2"]["out"][1],
        "K_TEMPORAL": temporal.shape[2] if net["order"] == "original" else temporal.shape[1],
        "BUF_A_ELEMS": a,
        "BUF_B_ELEMS": b,
    }
    lines = ["/* Generated by mibci.mcu; do not edit. */", "#pragma once"]
    return "\n".join(lines + [f"#define {k} {v}" for k, v in values.items()]) + "\n"


def write_headers(directory: Path, net: dict, rows: list[dict], trials: np.ndarray, precision: str) -> None:
    """dims.h, weights.h and trials.h for mcu/eegnet_int8.c (a quantized network from `quantize`) or
    mcu/eegnet_f32.c (an exported float network), with the input trials in the network's layout."""
    directory.mkdir(parents=True, exist_ok=True)
    n_classes = net["layers"][-1]["w"].shape[0]
    head = ["/* Generated by mibci.mcu; do not edit. */", "#pragma once", "#include <stdint.h>"]
    body = []
    if precision == "int8":
        for q in net["layers"]:
            name = q["name"]
            if q["op"] == "elu":
                body.append(_c_array("int8_t", f"{name}_table", q["table"]))
            if q["op"] in ("elu", "pool"):
                continue
            body.append(f"#define {name.upper()}_IN_OFFSET ({-q['in_zero']})")
            body.append(f"#define {name.upper()}_OUT_OFFSET ({q['out_zero']})")
            body.append(_c_array("int8_t", f"{name}_w", q["w"]))
            body.append(_c_array("int32_t", f"{name}_b", q["b"]))
            if q["op"] == "fc":
                body.append(f"#define FC_MULTIPLIER ({int(q['multiplier'][0])})")
                body.append(f"#define FC_SHIFT ({int(q['shift'][0])})")
            else:
                body.append(_c_array("int32_t", f"{name}_multiplier", q["multiplier"]))
                body.append(_c_array("int32_t", f"{name}_shift", q["shift"]))
        ctype, flat, out_bytes = "int8_t", trials.reshape(len(trials), -1), 1
    elif precision == "f32":
        for layer in net["layers"]:
            if "w" in layer:
                body.append(_c_array("float", f"{layer['name']}_w", layer["w"].astype(np.float32), per_line=8))
                body.append(_c_array("float", f"{layer['name']}_b", layer["b"].astype(np.float32), per_line=8))
        ctype, flat, out_bytes = "float", trials.reshape(len(trials), -1).astype(np.float32), 4
    else:
        raise ValueError("precision must be int8 or f32")
    (directory / "dims.h").write_text(_dims_header(net, rows, n_classes, out_bytes))
    (directory / "weights.h").write_text("\n".join(head + body) + "\n")
    rows_c = ",\n".join("    {" + ", ".join(_c_value(ctype, v) for v in t) + "}" for t in flat)
    (directory / "trials.h").write_text(
        "\n".join(head)
        + f"\n#define N_TRIALS {len(flat)}\n"
        + f'__attribute__((section(".trials"))) static const {ctype} trials[N_TRIALS][{flat.shape[1]}] = {{\n'
        + rows_c
        + "\n};\n"
    )
