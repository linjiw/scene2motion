"""Rebuild a descriptive recovery-displacement curve from the frozen A5 audit."""

import argparse
import json
from pathlib import Path

from experiments.astra_a4_adaptive_hold import save_or_check
from scene2motion.lean_supervisor import digest
from scene2motion.recovery_sensitivity import sensitivity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    for key in ("source_hashes", "analysis_source_hashes"):
        for path, expected in source[key].items():
            if digest(path) != expected:
                raise ValueError("upstream recovery source changed")
    root = Path(__file__).resolve().parents[1]
    result = {**sensitivity(source["rows"]), "input_sha256": digest(args.source),
        "analysis_source_hashes": {str(root / p): digest(root / p) for p in (
            "scene2motion/recovery_sensitivity.py", "experiments/analyze_astra_a5_progress_sensitivity.py")}}
    save_or_check(args.out, result)
    print(json.dumps({**result, "curve": [{k: v for k, v in r.items() if k != "by_unit"} for r in result["curve"]]}, indent=2))


if __name__ == "__main__":
    main()
