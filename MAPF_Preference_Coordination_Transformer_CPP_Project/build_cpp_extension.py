from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mapf_pct.cpp.generator import _load_extension


module = _load_extension()
print(f"C++ extension: OK ({Path(module.__file__).resolve()})")
