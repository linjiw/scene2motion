#!/usr/bin/env python3
"""Build the validated Scene2Motion-DB corpus-pilot release preview."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scene2motion.dataset_release import build_preview


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=ROOT / "outputs/corpus_pilot_v2")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "outputs/scene2motion_db_preview_v1")
    parser.add_argument("--include-clips", action="store_true",
                        help="copy the 268 motion payloads after redistribution review")
    args = parser.parse_args()
    receipt = build_preview(args.source, args.out, include_clips=args.include_clips)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
