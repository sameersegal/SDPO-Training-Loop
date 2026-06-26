"""Path resolution that works in BOTH layouts:

- local repo:  src/*.py, data/*.json, ojbench_data/ at repo root
- Modal container: everything flat at /root/app/ (shipped + volume-mounted)

So code finds its committed data files and the OJBench test cases regardless of
where it runs. Outputs are written to the CURRENT WORKING DIRECTORY by the
callers (so running from runs/iteration-XX/ keeps artifacts per-iteration).
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def repo_root() -> Path:
    """Repo root locally; the flat dir in the container."""
    p = HERE
    for _ in range(4):
        if (p / ".git").exists() or (p / "data" / "ojb_splits.json").exists():
            return p
        p = p.parent
    return HERE  # container: src files are flat, no repo markers


def find_file(name: str) -> Path:
    """Locate a committed data file (e.g. ojb_splits.json) across both layouts."""
    for base in (HERE, HERE.parent, repo_root(), repo_root() / "data"):
        c = base / name
        if c.exists():
            return c
    return HERE / name


def ojbench_dir() -> Path:
    """The OJBench test-case tree (contains NOI/loj-<id>/...)."""
    for base in (HERE, HERE.parent, repo_root()):
        c = base / "ojbench_data"
        if c.exists():
            return c
    return HERE / "ojbench_data"


def load_env() -> None:
    """Load .env (WANDB_API_KEY etc.) from the repo root if present.

    In the Modal container there is no .env — env vars come from Secrets — so this
    is a no-op there, which is correct.
    """
    for base in (repo_root(), HERE, HERE.parent):
        f = base / ".env"
        if f.exists():
            for line in f.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return
