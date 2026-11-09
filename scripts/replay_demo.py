"""Real-time replay of recorded EEG through the frozen decoder.

Streams one motor-imagery run at its original 250 Hz, band-passing it causally block
by block exactly as in training (zero filter state at the start of the run), and
decodes each trial from the window 0.5-2.5 s after the cue with the final ONNX model.
The display shows the scrolling EEG with the observation window, a cursor moved by
each decoded class, and the timing: the window closes 2.5 s after the cue and the
decision follows after the measured compute time.

This is a replay: the recorded participant could not see or react to the feedback,
so it does not validate online closed-loop control or asynchronous intent detection.

By default it replays session 2 with the frozen final configuration (seed 0), which
is only allowed once results/frozen_config.json exists. --save writes an MP4 that
plays in real time; without it the replay runs live in a window.
"""

import argparse
import json
import time
from pathlib import Path

import imageio_ffmpeg
import matplotlib
import numpy as np
from scipy.io import loadmat
from scipy.signal import sosfilt

from mibci import data, deploy
from mibci.experiment import MODELS_DIR, RESULTS_DIR
from mibci.model import load_bundle

FPS = 25
BLOCK = data.SFREQ // FPS  # samples streamed per frame (40 ms)
SHOW_S = 6.0  # seconds of EEG on screen
MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # left hand, right hand, feet, tongue
LABELS = ["left hand", "right hand", "feet", "tongue"]
ARROWS = ["←", "→", "↓", "↑"]
TRIALS_PER_RUN = 48


def load_run(subject: int, session: int, run: int):
    """Continuous raw EEG (samples x 22, microvolts) of one motor-imagery run, its
    trial-start samples, labels (0-3) and expert artifact flags."""
    mat = loadmat(data.download(subject)[session], struct_as_record=False, squeeze_me=True)["data"]
    r = [r for r in np.atleast_1d(mat) if np.size(r.trial) > 0][run]
    return (np.asarray(r.X, dtype=np.float64)[:, : len(data.EEG_CHANNELS)],
            np.atleast_1d(r.trial).astype(int) - 1, np.atleast_1d(r.y).astype(int) - 1,
            np.atleast_1d(r.artifacts).astype(bool))


class Replay:
    """Streams a run through the causal filter and decodes each trial when its window closes."""

    def __init__(self, raw, starts, bundle_path: Path, onnx_path: Path, offline_windows: np.ndarray):
        _, self.scaler, bundle = load_bundle(bundle_path)
        self.channels = bundle["channels"]
        self.raw = raw[:, self.channels]
        self.sos = data.bandpass_sos()
        self.zi = np.zeros((self.sos.shape[0], 2, len(self.channels)))
        self.filtered = np.zeros_like(self.raw)
        self.t = 0
        self.decide_at = starts + data.WIN_START + data.N_TIMES  # 2.5 s after each cue
        self.offline = offline_windows[:, self.channels]
        self.session = deploy.session(onnx_path)
        for _ in range(50):  # warm up, as a running device would be
            self.session.run(None, {deploy.INPUT: np.zeros((1, len(self.channels), data.N_TIMES), np.float32)})
        self.decisions = {}  # trial -> (predicted class, compute ms)
        self.max_mismatch = 0.0

    def step(self, n: int) -> list[int]:
        end = min(self.t + n, len(self.raw))
        if end <= self.t:
            return []
        self.filtered[self.t : end], self.zi = sosfilt(self.sos, self.raw[self.t : end], axis=0, zi=self.zi)
        self.t = end
        decided = []
        for k in np.flatnonzero(self.decide_at <= self.t):
            if k in self.decisions:
                continue
            window = self.filtered[self.decide_at[k] - data.N_TIMES : self.decide_at[k]].T
            # The streamed window must equal the window the decoder was evaluated on offline.
            self.max_mismatch = max(self.max_mismatch, float(np.abs(window - self.offline[k]).max()))
            x = self.scaler.transform(window[None])
            start = time.perf_counter()
            logits = self.session.run(None, {deploy.INPUT: x})[0]
            self.decisions[k] = (int(logits.argmax()), (time.perf_counter() - start) * 1e3)
            decided.append(k)
        return decided


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--session", type=int, default=2, choices=[1, 2])
    parser.add_argument("--run", type=int, default=0, help="motor-imagery run, 0-5")
    parser.add_argument("--trials", type=int, default=10, help="number of trials to replay")
    parser.add_argument("--bundle", type=Path, help="model bundle (default: frozen final configuration, seed 0)")
    parser.add_argument("--int8", action=argparse.BooleanOptionalAction, default=None,
                        help="use the INT8 model (default: the frozen configuration's precision)")
    parser.add_argument("--save", type=Path, help="write an MP4 instead of showing a window")
    args = parser.parse_args()

    frozen_path = RESULTS_DIR / "frozen_config.json"
    if args.session == 2 and not frozen_path.exists():
        raise SystemExit("Session 2 can only be replayed after the analysis plan is frozen.")
    final = json.loads(frozen_path.read_text())["final_configuration"] if frozen_path.exists() else None
    bundle = args.bundle or MODELS_DIR / "final" / f"sub-{args.subject:02d}_seed-0_{final['condition']}.pt"
    use_int8 = args.int8 if args.int8 is not None else bool(final and final["precision"] == "int8")
    onnx_path = bundle.with_name(f"{bundle.stem}_int8.onnx") if use_int8 else bundle.with_suffix(".onnx")

    raw, starts, labels, flags = load_run(args.subject, args.session, args.run)
    cached = data.load_session(args.subject, args.session, final_evaluation=args.session == 2)
    offline = cached["X"][args.run * TRIALS_PER_RUN : (args.run + 1) * TRIALS_PER_RUN]
    replay = Replay(raw, starts, bundle, onnx_path, offline)
    n_trials = min(args.trials, len(starts))
    end_sample = min(len(raw), int(replay.decide_at[n_trials - 1] + data.SFREQ))

    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    names = [data.EEG_CHANNELS[c] for c in replay.channels]
    shown = [names.index(c) for c in ("C3", "Cz", "C4") if c in names] or list(range(min(3, len(names))))
    fig = plt.figure(figsize=(12, 6.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[2.6, 1.4], height_ratios=[4, 1.2])
    ax_eeg, ax_cur, ax_hud = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
    precision = "INT8" if use_int8 else "FP32"
    fig.suptitle(f"REAL-TIME REPLAY of recorded EEG: participant {args.subject}, session {args.session}, "
                 f"run {args.run + 1} (the participant could not see this feedback)", fontsize=11, fontweight="bold")

    spacing = 60.0  # microvolts between traces
    lines = [ax_eeg.plot([], [], lw=0.8, color="#1f4e79")[0] for _ in shown]
    ax_eeg.set_xlim(-SHOW_S, 0)
    ax_eeg.set_ylim(-spacing * 0.8, spacing * (len(shown) - 0.2))
    ax_eeg.plot([-0.15, -0.15], [-spacing * 0.7, -spacing * 0.7 + 20], color="black", lw=2)
    ax_eeg.text(-0.2, -spacing * 0.7 + 10, "20 \u00b5V", ha="right", va="center", fontsize=8)
    ax_eeg.set_yticks([spacing * i for i in range(len(shown))], [names[i] for i in shown])
    ax_eeg.set_xlabel("time relative to now (s)")
    ax_eeg.set_title(f"Causally filtered EEG (4-38 Hz), {len(names)} of 22 electrodes used by the decoder", fontsize=9)
    overlays = []

    ax_cur.set_xlim(-6, 6)
    ax_cur.set_ylim(-6, 6)
    ax_cur.set_xticks([])
    ax_cur.set_yticks([])
    ax_cur.set_title("Cursor driven by the decoded class", fontsize=9)
    trail = ax_cur.plot([0], [0], color="grey", lw=1, alpha=0.6)[0]
    cursor = ax_cur.plot([0], [0], "o", ms=14, color="#1f4e79")[0]
    cue_text = ax_cur.text(0, 4.4, "", ha="center", fontsize=12)
    for cls, (dx, dy) in MOVES.items():
        ax_cur.text(5.6 * dx, 5.6 * dy, f"{ARROWS[cls]} {LABELS[cls]}", ha="center", va="center", fontsize=8,
                    color="grey", rotation=90 if dx else 0)
    ax_hud.axis("off")
    hud = ax_hud.text(0.0, 1.0, "", va="top", family="monospace", fontsize=9.5, transform=ax_hud.transAxes)
    path = [(0, 0)]

    def draw():
        t = replay.t
        lo = max(0, t - int(SHOW_S * data.SFREQ))
        x = (np.arange(lo, t) - t) / data.SFREQ
        for line, i, offset in zip(lines, shown, range(len(shown))):
            line.set_data(x, replay.filtered[lo:t, i] + spacing * offset)
        for artist in overlays:
            artist.remove()
        overlays.clear()
        current = None
        for k, s in enumerate(starts[:n_trials]):
            cue, w0 = s + int(data.CUE_ONSET_S * data.SFREQ), s + data.WIN_START
            if s <= t and (t - s) / data.SFREQ < 8.5:
                current = k
            if w0 < t and w0 + data.N_TIMES > lo:
                overlays.append(ax_eeg.axvspan((w0 - t) / data.SFREQ, (min(w0 + data.N_TIMES, t) - t) / data.SFREQ,
                                               color="#f4b183", alpha=0.35, lw=0))
            if lo < cue <= t:
                overlays.append(ax_eeg.axvline((cue - t) / data.SFREQ, color="#c00000", lw=1, ls="--"))
                overlays.append(ax_eeg.text((cue - t) / data.SFREQ, spacing * (len(shown) - 0.35), " cue",
                                            color="#c00000", fontsize=8))
            decide = replay.decide_at[k]
            if lo < decide <= t and k in replay.decisions:
                ok = replay.decisions[k][0] == labels[k]
                overlays.append(ax_eeg.axvline((decide - t) / data.SFREQ, color="#2e7d32" if ok else "#c62828", lw=2))
                overlays.append(ax_eeg.text((decide - t) / data.SFREQ, spacing * (len(shown) - 0.35),
                                            f" decision: {LABELS[replay.decisions[k][0]]}", fontsize=8,
                                            color="#2e7d32" if ok else "#c62828"))
        cue_text.set_text("" if current is None or t < starts[current] + 2 * data.SFREQ
                          else f"cue: {ARROWS[labels[current]]} {LABELS[labels[current]]}")

        done = [k for k in replay.decisions if k < n_trials]
        correct = sum(replay.decisions[k][0] == labels[k] for k in done)
        lines_hud = [
            "observation window: 0.5-2.5 s after the cue (2.0 s, shaded)   decision ready: cue + 2.5 s + compute",
            f"decoder: EEGNet, {len(names)} electrodes, {precision}, ONNX Runtime on CPU (1 thread)",
        ]
        if done:
            k = max(done)
            pred, ms = replay.decisions[k]
            lines_hud.append(f"trial {k + 1}: cue {LABELS[labels[k]]:<10} decoded {LABELS[pred]:<10} "
                             f"{'correct' if pred == labels[k] else 'wrong':<8} compute {ms:.2f} ms")
            lines_hud.append(f"running accuracy: {correct}/{len(done)} = {correct / len(done):.0%} (chance 25%)")
        hud.set_text("\n".join(lines_hud))

    def on_decisions(decided):
        for k in decided:
            pred = replay.decisions[k][0]
            dx, dy = MOVES[pred]
            x0, y0 = path[-1]
            path.append((float(np.clip(x0 + dx, -5, 5)), float(np.clip(y0 + dy, -5, 5))))
            cursor.set_data([path[-1][0]], [path[-1][1]])
            cursor.set_color("#2e7d32" if pred == labels[k] else "#c62828")
            trail.set_data(*zip(*path))

    if args.save:
        replay.step(max(0, starts[0] - data.SFREQ))  # stream (without drawing) up to 1 s before the first trial
        n_frames = (end_sample - replay.t) // BLOCK

        def frame(_):
            on_decisions(replay.step(BLOCK))
            draw()

        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        anim = animation.FuncAnimation(fig, frame, frames=n_frames, interval=1000 / FPS, repeat=False)
        anim.save(args.save, writer=animation.FFMpegWriter(fps=FPS, bitrate=1500), dpi=110)
    else:
        clock = time.perf_counter()

        def frame(_):
            target = min(end_sample, int((time.perf_counter() - clock) * data.SFREQ))  # follow the wall clock
            on_decisions(replay.step(max(0, target - replay.t)))
            draw()

        anim = animation.FuncAnimation(fig, frame, interval=1000 / FPS, cache_frame_data=False)
        plt.show()

    decided = [k for k in sorted(replay.decisions) if k < n_trials]
    acc = np.mean([replay.decisions[k][0] == labels[k] for k in decided]) if decided else float("nan")
    ms = [replay.decisions[k][1] for k in decided]
    print(f"{len(decided)} trials decoded, accuracy {acc:.2f}, compute median {np.median(ms):.3f} ms; "
          f"streamed windows match the offline windows to {replay.max_mismatch:.1e} uV")


if __name__ == "__main__":
    main()
