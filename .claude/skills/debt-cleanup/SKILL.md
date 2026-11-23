---
name: debt-cleanup
description: Pay down old code-quality debt in the COMP 690 mibci project without changing any result. Use when asked to clean up lint debt, "garbage-collect" old problems, shrink tools/lint_baseline.json, or refactor a script while keeping its outputs byte-identical.
---

# Debt cleanup (garbage collection for old slop)

New violations are already blocked by `python3 tools/check.py`: a file can never gain a
violation of a rule that `tools/lint_baseline.json` does not already allow. This skill is the
other half, removing the violations that were there before the ratchet existed. Every change
must leave every published number, table, figure and report byte-identical.

## 1. Pick a batch

```bash
python3 tools/check.py --todo
```

Take one file, or a few files that one generator covers. Never touch a file the guard calls
pinned (`tools/guard.py`, check `pinned-source`). Finished studies record those files'
SHA-256, and `ruff` is configured to skip them.

## 2. Know what proves "nothing changed"

| File kind | Proof |
| --- | --- |
| Report, slide, analysis or manifest generator | `python3 tools/golden.py <name>` prints SAME (see `--list`) |
| Library code in `src/mibci` | its unit tests, plus golden for the generators that import it |
| Training/evaluation pipeline scripts | unit tests; keep the change syntactic and say so |
| PDF export | a throwaway `uv venv` with reportlab and pillow; compare page count and extracted text |

Run the proof before you edit, so you know it is green to begin with.

## 3. Fix

- Fix the cause, not the message. A long prose line becomes adjacent string literals. A
  lambda bound to a name becomes a `def`. `zip()` gets `strict=True`: equal lengths are an
  invariant, so a mismatch should raise, not truncate quietly.
- If golden output changes, the edit changed behaviour. Undo it, or stop and report why the
  old behaviour was a bug.
- A `# noqa: CODE` needs a reason on the same line, and only when the rule is wrong for that
  line (for example `numpy.load` on an NPZ the script itself wrote).
- Do not rename public functions, CLI flags, output paths or JSON keys. Other scripts, docs
  and archived records depend on them.
- Format only your own files: `uvx ruff@0.17.6 format <files>` and
  `uvx ruff@0.17.6 check --fix <files>`.

## 4. Lock in the gain

```bash
python3 tools/check.py --files <your files>   # format and lint for your files
python3 tools/check.py --fast                 # whole tree, no unit tests
python3 tools/check.py --tighten              # lower the allowed counts; it never raises them
```

`--tighten` refuses to run while any count rose, so a cleanup can never smuggle in new debt.

## 5. Report

List each file with its violation count before and after, the proof you ran and its output
(SAME/DIFF, tests passed), and anything you left alone and why.

Not in scope: changing a method, a statistic, a default, a dependency or uv.lock, and
re-running training. Those are research changes and need their own study (see AGENTS.md).
