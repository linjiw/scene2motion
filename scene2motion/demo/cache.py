"""Content-addressed clip cache with provenance.

The demo must be able to say where a motion came from. A cache entry therefore stores not
just the qpos array but the exact inputs that produced it -- scene parameters, preference,
the constraint program's own bytes, the model name, seed and denoising steps -- so a hit is
verifiable rather than merely fast, and a stale entry cannot silently survive a change to the
planner or the program encoding.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

# v2 invalidates clips made by runner noise-stream v1, which restarted each per-sample seed at
# every autoregressive window. Reusing those clips after the runner fix would silently mix two
# different stochastic generators in one comparison.
CACHE_VERSION = 2


def key_for(*, scene_id: str, preference: str, program_bytes: bytes, model: str,
            seed: int, steps: int, fps: float, n_frames: int) -> str:
    h = hashlib.sha1()
    h.update(f"v{CACHE_VERSION}|{scene_id}|{preference}|{model}|{seed}|{steps}|"
             f"{fps:.4f}|{n_frames}|".encode())
    h.update(program_bytes)
    return h.hexdigest()[:20]


class ClipCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.root / f"{key}.npy", self.root / f"{key}.json"

    def get(self, key: str) -> tuple[np.ndarray, dict] | None:
        qp, mp = self._paths(key)
        if not (qp.exists() and mp.exists()):
            return None
        try:
            return np.load(qp), json.loads(mp.read_text())
        except Exception:
            return None

    def put(self, key: str, qpos: np.ndarray, meta: dict) -> None:
        qp, mp = self._paths(key)
        np.save(qp, np.asarray(qpos, np.float32))
        mp.write_text(json.dumps({**meta, "key": key, "cached_at": time.time(),
                                  "cache_version": CACHE_VERSION}, indent=2))

    def stats(self) -> dict:
        keys = sorted(p.stem for p in self.root.glob("*.npy"))
        size = sum(p.stat().st_size for p in self.root.glob("*.npy"))
        return {"n_entries": len(keys), "bytes": size, "keys": keys[:64]}
