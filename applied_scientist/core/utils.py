from __future__ import annotations

import os
import tempfile


def atomic_write(path: str, content: str) -> None:
    """Write atomically: write to temp file, then rename (POSIX atomic)."""
    dir_name = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def normalize_name(name: str) -> str:
    """Normalize experiment name for dedup comparison.
    'mappo_v2' and 'MAPPO-v2' both become 'mappo_v2'."""
    return name.lower().replace("-", "_").strip()
