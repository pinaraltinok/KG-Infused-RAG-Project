from __future__ import annotations

import os
import re
from pathlib import Path


def load_env_file(path: Path | str = ".env") -> None:
    """
    Minimal .env loader supporting:
      - KEY=VALUE
      - PowerShell style: $env:KEY = "VALUE"

    Existing environment variables are not overwritten.
    """

    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.exists():
        return

    ps_re = re.compile(r"^\s*\$env:(?P<k>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<v>.+?)\s*$")
    kv_re = re.compile(r"^\s*(?P<k>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<v>.+?)\s*$")

    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        m = ps_re.match(line) or kv_re.match(line)
        if not m:
            continue

        k = m.group("k")
        v = m.group("v").strip()

        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]

        os.environ.setdefault(k, v)

