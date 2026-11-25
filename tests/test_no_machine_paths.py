"""No machine-specific absolute paths in tracked files: locations come from CSN_DATA_ROOT, CSN_RUNS,
CSN_EXP_OUT, CSN_ACTIVATE or command-line arguments.  Run: pytest tests/  (or python tests/test_no_machine_paths.py)"""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATTERN = r"/vast/|/gpfs/|/nfs/|/Users/|/home/[a-z]{2,}[0-9]+"


def offending_lines():
    r = subprocess.run(["git", "grep", "-nIE", PATTERN, "--", ".", ":!tests/test_no_machine_paths.py"],
                       cwd=REPO, capture_output=True, text=True)
    return [l for l in r.stdout.splitlines() if l]


def test_no_machine_paths():
    bad = offending_lines()
    assert not bad, "machine-specific paths:\n" + "\n".join(bad)


if __name__ == "__main__":
    bad = offending_lines()
    print("\n".join(bad) or "ok")
    raise SystemExit(1 if bad else 0)
