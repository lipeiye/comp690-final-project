"""Run the project's heavy studies on UNC Longleaf (and only Longleaf; never Sycamore).

Access: `ssh longleaf`, the host alias in ~/.ssh/config, logs in with the key
~/.ssh/id_ed25519_longleaf installed in the Longleaf account: no password and no Duo, for any
agent on this Mac, as long as the UNC VPN is connected. The project lives in the Longleaf
home directory, because the account has no /work space; the home quota is 50 GB, so large
raw downloads are deleted once cached.

    python3 tools/longleaf.py check                  # is the connection open? where is the project?
    python3 tools/longleaf.py push [--data]          # code and results up (--data: EEG cache and final models)
    python3 tools/longleaf.py setup                  # uv and the locked environment on Longleaf
    python3 tools/longleaf.py submit region_study --stage train --part cv --tasks 199 --cpus 4 --hours 4
    python3 tools/longleaf.py submit region_study --stage evaluate --tasks 51 --after 10041234
    python3 tools/longleaf.py submit prepare_lee2019 --plain --partition datamover --hours 12 -- --delete-raw
    python3 tools/longleaf.py status
    python3 tools/longleaf.py pull region_study      # that study's records back into results/supplementary/

`submit` writes a SLURM array script: task i of n runs `scripts/<study>.py <args> --shard i/n
--workers <cpus>` with MIBCI_DEVICE=cpu, one torch thread per worker. Studies skip finished
jobs, so resubmitting the same command only runs what is missing. Task i takes every n-th job
of the whole plan, so pick an n that shares no factor with the plan's repeating pattern: the
region plan repeats every 4 jobs (3 folds, then the final model), and with n = 60 every final
model landed in the same 15 tasks, which ran out of time on 2026-11-21. Odd n spreads them.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST = "longleaf"  # ~/.ssh/config alias: key login and a shared connection
SSH = ["ssh", "-o", "BatchMode=yes"]
REMOTE = "comp690"  # relative to the Longleaf home directory (ssh and rsync start there)
# every folder a study plan pins files from; kernels/ (the C sources int8_kernel_x86_study compiles) and mcu/
# were missing until 2026-11-22 (tests/test_longleaf.py now checks this)
CODE = [
    "src",
    "scripts",
    "tests",
    "tools",
    "kernels",
    "mcu",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "README.md",
]
BASE = f"$HOME/{REMOTE}"  # expanded by the shell on Longleaf
ENV = f"UV_PROJECT_ENVIRONMENT={BASE}/.venv MIBCI_DATA_DIR={BASE}/data MIBCI_RESULTS_DIR={BASE}/results"


def ssh(command: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([*SSH, HOST, command], check=check, text=True)


def rsync(sources: list[str], target: str, extra: tuple[str, ...] = ()) -> None:
    command = ["rsync", "-az", "--exclude", "__pycache__", "--exclude", ".DS_Store", *extra,
               "-e", " ".join(SSH), *sources, target]  # fmt: skip
    subprocess.run(command, check=True, cwd=ROOT)


def check(_args) -> None:
    probe = subprocess.run([*SSH, "-o", "ConnectTimeout=15", HOST, "true"], capture_output=True, text=True, check=False)
    if probe.returncode != 0:
        raise SystemExit(f"Cannot reach Longleaf ({probe.stderr.strip()}). Is the UNC VPN connected?")
    ssh(f"hostname; mkdir -p {REMOTE} && cd {REMOTE} && pwd; quota -s | tail -2; sinfo -s | head -12")


def push(args) -> None:
    ssh(f"mkdir -p {REMOTE}/results {REMOTE}/data")
    rsync(CODE, f"{HOST}:{REMOTE}/")  # macOS ships openrsync: keep to flags it supports
    # results: never overwrite a record that a Longleaf job wrote
    rsync(["results/"], f"{HOST}:{REMOTE}/results/", ("--ignore-existing",))
    if args.data:
        rsync(["data.nosync/processed"], f"{HOST}:{REMOTE}/data/")
        rsync(["data.nosync/models/final"], f"{HOST}:{REMOTE}/data/models/", ("--ignore-existing",))


def setup(_args) -> None:
    ssh(
        "set -e; export PATH=$HOME/.local/bin:$PATH; "
        "command -v uv >/dev/null || python3 -m pip install --user --quiet uv; "
        f"cd {REMOTE} && {ENV} uv sync --locked && "
        f"{ENV} .venv/bin/python -m unittest discover -s tests 2>&1 | tail -3"
    )


def submit(args) -> None:
    """A SLURM array over a study's shards, or with --plain one task running the script as given."""
    if args.plain:
        name, study_args, sharding = f"{args.study}", " ".join(args.extra), ""
        args.tasks = 1
    else:
        if not args.stage:
            raise SystemExit("--stage is required unless --plain")
        name = f"{args.study}-{args.stage}" + (f"-{args.part}" if args.part else "")
        study_args = (
            f"--stage {args.stage}" + (f" --part {args.part}" if args.part else "") + " " + " ".join(args.extra)
        )
        sharding = f" --shard $SLURM_ARRAY_TASK_ID/{args.tasks} --workers {args.cpus}"
    # node features go on the sbatch command line: Longleaf ignored a --constraint line inside the script
    options = f"--constraint={shlex.quote(args.constraint)} " if args.constraint else ""
    options += f"--exclude={shlex.quote(args.exclude)} " if args.exclude else ""
    # start only when that job has ended (any outcome): a stage that needs its inputs checks for them itself
    after = ":".join(str(int(j)) for j in args.after.split(":")) if args.after else ""  # one id, or several as a:b
    options += f"--dependency=afterany:{after} " if after else ""
    if args.constraint:
        name += "-" + args.constraint.replace("&", "-")
    script = f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition={args.partition}
#SBATCH --array=0-{args.tasks - 1}%{args.concurrent}
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --mem={args.mem}
#SBATCH --time={args.hours}:00:00
#SBATCH --output={REMOTE}/logs/{name}-%A_%a.out
set -euo pipefail
cd {BASE}
export {ENV} MIBCI_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python scripts/{args.study}.py {study_args}{sharding}
"""
    path = f"{REMOTE}/logs/{name}.sbatch"
    ssh(f"mkdir -p {REMOTE}/logs && cat > {path} <<'EOF'\n{script}EOF\nsbatch {options}{path}")


def status(_args) -> None:
    ssh(f"squeue -u $USER -o '%.18i %.30j %.8T %.10M %.6D %R' | head -40; "
        f"for d in {REMOTE}/results/supplementary/*/; do "
        "echo \"$(basename $d): $(find $d -name '*.json' | wc -l) records\"; done")  # fmt: skip


def pull(args) -> None:
    """Records and models that exist only on Longleaf; a file this Mac already has is never replaced.

    Records and models are written once, so a local copy is already the same file; the analysis
    outputs in a study directory are made here and only pushed up, so the Longleaf copies are
    older. On 2026-11-22 a plain pull would have put the 02:41 noise-mechanism summary over the
    03:01 one."""
    target = ROOT / "results/supplementary"
    keep = ("--ignore-existing",)
    rsync([f"{HOST}:{REMOTE}/results/supplementary/{args.study}"], f"{target}/", keep)
    if args.models:
        (ROOT / "data.nosync/models/supplementary").mkdir(parents=True, exist_ok=True)
        rsync([f"{HOST}:{REMOTE}/data/models/supplementary/{args.study}"], "data.nosync/models/supplementary/", keep)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check").set_defaults(run=check)
    p = sub.add_parser("push")
    p.add_argument("--data", action="store_true", help="also the EEG cache and the archived final models")
    p.set_defaults(run=push)
    sub.add_parser("setup").set_defaults(run=setup)
    p = sub.add_parser("submit")
    p.add_argument("study", help="script name in scripts/, without .py")
    p.add_argument("--stage", help="the study stage (required unless --plain)")
    p.add_argument("--plain", action="store_true", help="one task, no --stage/--shard/--workers added")
    p.add_argument("--part")
    p.add_argument("--tasks", type=int, default=100)
    p.add_argument("--concurrent", type=int, default=200, help="array tasks running at once")
    p.add_argument("--cpus", type=int, default=4)
    p.add_argument("--mem", default="8G")
    p.add_argument("--hours", type=int, default=4)
    p.add_argument("--partition", default="general")
    p.add_argument("--constraint", help="node features, e.g. 'intel&avx512' or 'amd'")
    p.add_argument("--exclude", help="nodes to avoid, e.g. 'c[0301-0317,0401-0414]'")
    p.add_argument(
        "--after", help="SLURM job id(s) that must end first, e.g. a train array before its evaluate stage; a:b for two"
    )
    p.set_defaults(run=submit)
    sub.add_parser("status").set_defaults(run=status)
    p = sub.add_parser("pull")
    p.add_argument("study")
    p.add_argument("--models", action="store_true", help="also copy the study's models")
    p.set_defaults(run=pull)
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1 :] if "--" in argv else []  # after "--": passed to the study script
    args = parser.parse_args(argv[: argv.index("--")] if "--" in argv else argv)
    args.extra = [shlex.quote(a) for a in extra]
    args.run(args)


if __name__ == "__main__":
    sys.exit(main())
