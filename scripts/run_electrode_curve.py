"""Second supplementary wave: ten seeds for every point of the electrode-count curve.

Planned on 2026-11-12, after the first wave (scripts/run_electrode_study.py) had been
scored. That wave gave ten seeds to 22, 16 and 8 electrodes but only three to 20, 18,
14, 12 and 10. With ten seeds the 16-electrode models score about 1.3 points above all
22, while their three-seed neighbours at 18 and 14 do not. One run moves by about 3.5
points with the seed alone, so three seeds cannot tell a peak at 16 from seed noise.
This wave adds seeds 3-9 for the five intermediate counts.

Nothing else changes. Subsets are the first k electrodes of the archived session-1
relevance order, and training and scoring call the first wave's own functions with
only the output folders redirected. The decision to run this wave was made after
looking at first-wave test results, so it is exploratory in the same sense: it
describes the curve and is not a basis for choosing an electrode count.

Stages: plan, train, evaluate (same options as run_electrode_study.py).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import run_electrode_study as wave1
from mibci import data
from mibci.artifacts import completed, model_exports, require_files, sha256, write_json
from mibci.env import environment
from mibci.experiment import MODELS_DIR, RESULTS_DIR, device_for, run_pool

OUT = RESULTS_DIR / "supplementary/electrode_count_wave2"
MODELS = MODELS_DIR / "supplementary/electrode_count_wave2"
FIRST = RESULTS_DIR / "supplementary/electrode_count/protocol.json"
CORE = [*wave1.CORE, "scripts/run_electrode_curve.py"]
WAVE = "curve"


def redirect() -> None:
    """Point the first wave's functions at this wave's folders. Called in every process."""
    wave1.OUT, wave1.MODELS = OUT, MODELS


def train_job(job: dict, epochs: int) -> None:
    redirect()
    wave1.train_job(job, epochs)


def evaluate_job(job: dict) -> None:
    redirect()
    wave1.evaluate_job(job)


def load_plan() -> dict:
    plan = json.loads((OUT / "protocol.json").read_text())
    checks = {**{data.PROJECT_ROOT / p: d for p, d in plan["source_code_sha256"].items()},
              **{RESULTS_DIR / p: d for p, d in plan["ranking_sha256"].items()},
              RESULTS_DIR / "frozen_config.json": plan["historical_frozen_config_sha256"],
              FIRST: plan["first_wave_protocol_sha256"]}
    for path, digest in checks.items():
        if sha256(path) != digest:
            raise RuntimeError(f"Pinned file changed: {path}; start a new supplementary directory")
    return plan


def create_plan() -> None:
    if (OUT / "protocol.json").exists():
        load_plan()
        print("Existing second-wave protocol retained.")
        return
    first = wave1.load_plan()  # also checks that the first wave's pins still hold
    sweep = {(j["subject"], j["condition"]): j["channels"] for j in first["jobs"] if j["wave"] == "sweep"}
    jobs = []
    for subject in data.SUBJECTS:
        order = json.loads((RESULTS_DIR / f"final/rankings/sub-{subject:02d}_seed-0.json").read_text())["orders"]["relevance"]
        for seed in first["new_seeds"]:
            for k in first["sweep_counts"]:
                condition, channels = f"relevance-{k}", sorted(int(c) for c in order[:k])
                if channels != sweep[(subject, condition)]:
                    raise RuntimeError(f"Subset differs from the first wave: subject {subject}, {condition}")
                jobs.append({"name": f"sub-{subject:02d}_seed-{seed}_{condition}", "subject": subject, "seed": seed,
                             "condition": condition, "channels": channels, "wave": WAVE})
    jobs.sort(key=lambda j: (j["seed"], j["condition"], j["subject"]))
    write_json(OUT / "protocol.json", {
        "created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "status": "supplementary_post_test",
        "historical_test_results_already_known": True,
        "first_wave_results_already_known": True,
        "purpose": "Give every point of the electrode-count curve ten seeds, to tell a peak at 16 electrodes "
                   "from seed noise",
        "selection": "All nine participants; first k electrodes of the archived session-1 relevance order; "
                     "no new selection and no change to the historical final configuration",
        "epochs": first["epochs"], "batch_size": first["batch_size"], "lr": first["lr"],
        "seeds": first["new_seeds"], "counts": first["sweep_counts"],
        "primary": first["primary"], "secondary": first["secondary"], "bootstrap": first["bootstrap"],
        "claim_boundary": "Exploratory. Planned after the historical test results and the first-wave results were "
                          "known. Describes the curve; not a basis for choosing an electrode count.",
        "historical_frozen_config_sha256": first["historical_frozen_config_sha256"],
        "first_wave_protocol_sha256": sha256(FIRST),
        "source_code_sha256": {p: sha256(data.PROJECT_ROOT / p) for p in CORE},
        "ranking_sha256": first["ranking_sha256"],
        "jobs": jobs,
    })
    print(f"Planned {len(jobs)} training runs: seeds {first['new_seeds']}, electrode counts {first['sweep_counts']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["plan", "train", "evaluate"], required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=data.SUBJECTS)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not set(args.subjects) <= set(data.SUBJECTS):
        parser.error("subjects must be in 1..9")
    if args.stage == "plan":
        create_plan()
        return
    plan = load_plan()
    redirect()
    selected = [j for j in plan["jobs"] if j["subject"] in args.subjects]
    bundle, record = wave1.bundle_path, wave1.record_path
    if args.stage == "train":
        write_json(OUT / "environments" / ("subjects-" + "-".join(map(str, args.subjects)) + ".json"),
                   {**environment(), "device_assignment": {str(s): device_for(s) for s in args.subjects}})
        jobs = [(j["name"], {"job": j, "epochs": plan["epochs"]}) for j in selected
                if not completed(record(j, "runs"), [bundle(j)],
                                 {"subject": j["subject"], "seed": j["seed"], "condition": j["condition"],
                                  "epochs": plan["epochs"]})]
        run_pool(train_job, jobs, args.workers, threads=1)
    else:
        require_files([bundle(j) for j in selected], "second-wave evaluation inputs")
        jobs = [(j["name"], {"job": j}) for j in selected
                if not completed(record(j, "test"), [bundle(j), *model_exports(bundle(j))], {"name": j["name"]})]
        run_pool(evaluate_job, jobs, args.workers, threads=1)


if __name__ == "__main__":
    main()
