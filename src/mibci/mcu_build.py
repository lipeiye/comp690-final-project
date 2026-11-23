"""Build and run the C networks of mcu/ for emulated Cortex-M cores (QEMU), and read what they report.

Tools are passed in or found on PATH, never assumed: the Arm GNU toolchain's bin directory (or
arm-none-eabi-gcc on PATH), a CMSIS-NN source tree, and qemu-system-arm on PATH, whose installation
prefix also supplies qemu-plugin.h and GLib's headers for the instruction-counting plugin (mcu/icount.c).

QEMU executes instructions but models no timing, so a run yields instruction counts, not cycles or
energy. Each core is a compiler target plus a board whose memory map the linker script receives:

  m0         Cortex-M0 (Armv6-M, no DSP, no FPU), built for and counted on mps2-an385; the same binary
             executes the same instructions on any M-profile core. Whether it fits a 16 KB part is
             tested by linking and running it on the micro:bit board (nRF51, 16 KB RAM)
  m4         Cortex-M4F (DSP extension, single-precision FPU) on mps2-an386
  m4-nodsp   int8 code for Cortex-M3 (Armv7-M without DSP) on the same M4 board: CMSIS-NN's plain-C
             kernels, to separate integer arithmetic from SIMD
  m55        Cortex-M55 with Helium (MVE integer and float) on mps3-an547
  m55-nomve  Cortex-M55 with Helium switched off (DSP and scalar FPU only)
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

from mibci.paths import PROJECT_ROOT

MCU_DIR = PROJECT_ROOT / "mcu"
KIB, MIB = 1024, 1024 * 1024
CORES = {
    "m0": {"flags": ["-mcpu=cortex-m0", "-mthumb", "-mfloat-abi=soft"], "board": "mps2-an385"},
    "m4": {"flags": ["-mcpu=cortex-m4", "-mthumb", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard"], "board": "mps2-an386"},
    "m4-nodsp": {"flags": ["-mcpu=cortex-m3", "-mthumb", "-mfloat-abi=soft"], "board": "mps2-an386"},
    "m55": {"flags": ["-mcpu=cortex-m55", "-mthumb", "-mfloat-abi=hard"], "board": "mps3-an547"},
    "m55-nomve": {"flags": ["-mcpu=cortex-m55+nomve", "-mthumb", "-mfloat-abi=hard"], "board": "mps3-an547"},
}
# (origin, length) of FLASH, TRIALS and RAM in QEMU's model of each board
BOARDS = {
    "microbit": {"flash": (0x0, 192 * KIB), "trials": (0x30000, 64 * KIB), "ram": (0x20000000, 16 * KIB)},
    "mps2-an385": {"flash": (0x0, 4 * MIB), "trials": (0x21000000, 16 * MIB), "ram": (0x20000000, 4 * MIB)},
    "mps2-an386": {"flash": (0x0, 4 * MIB), "trials": (0x21000000, 16 * MIB), "ram": (0x20000000, 4 * MIB)},
    "mps3-an547": {"flash": (0x0, 512 * KIB), "trials": (0x21000000, 4 * MIB), "ram": (0x20000000, 512 * KIB)},
}
LIBRARY_FLAGS = ["-Ofast", "-ffunction-sections", "-fdata-sections", "-DARM_NN_ENABLE_F32=1"]  # CMSIS-NN default
PROGRAM_FLAGS = ["-O2", "-std=c11", "-ffunction-sections", "-fdata-sections", "-DARM_NN_ENABLE_F32=1", "-Wall"]
SOURCES = {"int8": "eegnet_int8.c", "f32": "eegnet_f32.c"}
FLASH_SECTIONS = (".text", ".ARM.extab", ".ARM.exidx", ".init_array", ".fini_array", ".data")


class BuildError(RuntimeError):
    pass


def tools(toolchain: Path | None, cmsis: Path | None) -> dict[str, Path]:
    """Paths of the cross compiler, its binutils, QEMU and CMSIS-NN; stops if one is missing."""
    gcc = Path(toolchain) / "arm-none-eabi-gcc" if toolchain else shutil.which("arm-none-eabi-gcc")
    qemu = shutil.which("qemu-system-arm")
    missing = [name for name, path in (("arm-none-eabi-gcc", gcc), ("qemu-system-arm", qemu), ("CMSIS-NN", cmsis))
               if not path or not Path(path).exists()]  # fmt: skip
    if missing:
        raise SystemExit(f"Missing {', '.join(missing)}: pass --toolchain and --cmsis-nn, put QEMU on PATH")
    gcc = Path(gcc)
    return {"gcc": gcc, "ar": gcc.with_name("arm-none-eabi-ar"), "size": gcc.with_name("arm-none-eabi-size"),
            "nm": gcc.with_name("arm-none-eabi-nm"), "qemu": Path(qemu), "cmsis": Path(cmsis)}  # fmt: skip


def versions(t: dict[str, Path]) -> dict[str, str]:
    def first_line(cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.splitlines()[0]

    commit = subprocess.run(
        ["git", "-C", str(t["cmsis"]), "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    return {"gcc": first_line([t["gcc"], "--version"]), "qemu": first_line([t["qemu"], "--version"]),
            "cmsis_nn_commit": commit.stdout.strip() or "unknown"}  # fmt: skip


def _run(cmd: list, cwd: Path | None = None) -> str:
    done = subprocess.run([str(c) for c in cmd], check=False, capture_output=True, text=True, cwd=cwd)
    if done.returncode != 0:
        raise BuildError(f"{Path(str(cmd[0])).name} failed:\n{done.stdout}{done.stderr}")
    return done.stdout + done.stderr


def plugin(t: dict[str, Path], build: Path) -> Path:
    """The instruction-counting plugin, compiled for the host against this QEMU's headers."""
    out = build / ("libicount.dylib" if sys.platform == "darwin" else "libicount.so")
    source = MCU_DIR / "icount.c"
    if out.exists() and out.stat().st_mtime > source.stat().st_mtime:
        return out
    prefix = t["qemu"].parent.parent  # the installation prefix on PATH (Homebrew keeps GLib there, not in QEMU's keg)
    includes = [prefix / "include", prefix / "include" / "glib-2.0", prefix / "lib" / "glib-2.0" / "include"]
    build.mkdir(parents=True, exist_ok=True)
    _run(["cc", "-O2", "-shared", "-fPIC", "-Wl,-undefined,dynamic_lookup", *[f"-I{i}" for i in includes],
          "-o", out, source])  # fmt: skip
    return out


def library(t: dict[str, Path], core: str, build: Path) -> Path:
    """CMSIS-NN compiled for one core (every source file; the linker keeps what is used)."""
    out = build / "lib" / core / "libcmsis-nn.a"
    if out.exists():
        return out
    objects = out.parent / "obj"
    objects.mkdir(parents=True, exist_ok=True)
    sources = sorted(  # what CMSIS-NN's CMake builds with ARM_NN_ENABLE_F32 on and F16 off
        p for p in (t["cmsis"] / "Source").rglob("*.c") if "Bindings" not in p.parts and "f16" not in p.stem
    )
    if len({p.stem for p in sources}) != len(sources):  # objects share one directory
        raise BuildError("duplicate CMSIS-NN source names")
    for i in range(0, len(sources), 40):  # one compiler call per batch of files
        batch = sources[i : i + 40]
        _run([t["gcc"], *CORES[core]["flags"], *LIBRARY_FLAGS, f"-I{t['cmsis'] / 'Include'}", "-c", *batch], objects)
    _run([t["ar"], "rcs", out, *sorted(objects.glob("*.o"))])
    return out


def linker_script(board: str, stack_bytes: int, path: Path) -> Path:
    regions = BOARDS[board]
    text = (MCU_DIR / "board.ld.in").read_text()
    for name, (origin, length) in regions.items():
        text = text.replace(f"@{name.upper()}_ORIGIN@", hex(origin)).replace(f"@{name.upper()}_LENGTH@", str(length))
    path.write_text(text.replace("@STACK_BYTES@", str(stack_bytes)))
    return path


def link(t, core: str, precision: str, headers: Path, out: Path, board: str, scratch: int, stack: int) -> Path:
    """Compile the harness and one network against generated headers and link for a board."""
    out.mkdir(parents=True, exist_ok=True)
    elf = out / "net.elf"
    lib = library(t, core, out.parent.parent / "_lib")
    script = linker_script(board, stack, out / "board.ld")
    sources = [MCU_DIR / s for s in ("main.c", "semihost.c", "startup.c", SOURCES[precision])]
    _run([t["gcc"], *CORES[core]["flags"], *PROGRAM_FLAGS, f"-DSCRATCH_BYTES={scratch}", f"-I{headers}",
          f"-I{MCU_DIR}", f"-I{t['cmsis'] / 'Include'}", *sources, lib, "--specs=nano.specs", "-nostartfiles",
          "-Wl,--gc-sections", f"-Wl,-Map,{out / 'net.map'}", "-T", script, "-lm", "-o", elf])  # fmt: skip
    return elf


def sections(t, elf: Path) -> dict[str, int]:
    rows = _run([t["size"], "-A", elf]).splitlines()
    sizes = {}
    for row in rows:
        parts = row.split()
        if len(parts) >= 2 and parts[0].startswith(".") and parts[1].isdigit():
            sizes[parts[0]] = int(parts[1])
    return sizes


def symbol(t, elf: Path, name: str) -> int:
    for row in _run([t["nm"], elf]).splitlines():
        parts = row.split()
        if len(parts) == 3 and parts[2] == name:
            return int(parts[0], 16)
    raise BuildError(f"{name} not found in {elf}")


def emulate(t, elf: Path, board: str, plugin_path: Path, timeout_s: int = 600) -> dict:
    """Run on QEMU; returns scratch bytes, per-trial output bytes and layer counts, stack use."""
    start, stop = symbol(t, elf, "mark_start"), symbol(t, elf, "mark_stop")
    cmd = [t["qemu"], "-M", board, "-nographic", "-monitor", "none", "-serial", "none",
           "-semihosting-config", "enable=on,target=native", "-kernel", elf, "-d", "plugin",
           "-plugin", f"{plugin_path},start={hex(start)},stop={hex(stop)}"]  # fmt: skip
    done = subprocess.run([str(c) for c in cmd], check=False, capture_output=True, text=True, timeout=timeout_s)
    text = done.stdout + done.stderr
    result = {"exit_status": done.returncode, "outputs": {}, "regions": []}
    for line in text.splitlines():
        if m := re.fullmatch(r"scratch( too small)? (\d+)", line):
            result["scratch_needed"], result["scratch_too_small"] = int(m.group(2)), bool(m.group(1))
        elif m := re.fullmatch(r"trial (\d+) ([0-9a-f]+)", line):
            result["outputs"][int(m.group(1))] = bytes.fromhex(m.group(2))
        elif m := re.fullmatch(r"stack (\d+)", line):
            result["stack_used"] = int(m.group(1))
        elif m := re.fullmatch(r"region \d+: (\d+) instructions", line):
            result["regions"].append(int(m.group(1)))
        elif m := re.fullmatch(r"total: (\d+) instructions", line):
            result["total_instructions"] = int(m.group(1))
        elif line.strip() == "fault":
            result["fault"] = True
    return result
