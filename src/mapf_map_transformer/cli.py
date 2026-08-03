from __future__ import annotations

import sys
from pathlib import Path


def _run_root_script(name: str) -> None:
    root = Path(__file__).resolve().parents[2]
    namespace = {"__name__": "__main__", "__file__": str(root / name)}
    code = compile((root / name).read_text(encoding="utf-8"), str(root / name), "exec")
    exec(code, namespace)


def train_main() -> None:
    _run_root_script("train.py")


def evaluate_main() -> None:
    _run_root_script("evaluate.py")
