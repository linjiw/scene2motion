"""Experiment C: which test results are interpolation and which are compositional extrapolation.

A single test MAE mixes two very different questions. The test split holds unseen beam HEIGHTS
and POSITIONS at beam counts the model trained on (interpolation), and it also holds THREE-beam
scenes, a count that appears nowhere in training (compositional extrapolation). Reporting one
number for both would let the easy half carry the hard half.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model_v3 import DemandOnly, DuckTCN
from .response import DIP_MAX


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="outputs/duck_dataset_v3_m018")
    ap.add_argument("--model", default="outputs/duck_model_v3_m018")
    args = ap.parse_args()
    root, mdir = Path(args.data), Path(args.model)
    dmeta = json.loads((root / "meta.json").read_text())
    mmeta = json.loads((mdir / "tcn.json").read_text())
    assert mmeta["dataset_hash"] == dmeta["dataset_hash"], "model/dataset hash mismatch"

    d = np.load(root / "test.npz")
    X = torch.from_numpy(d["X"]).float()
    R = torch.from_numpy(d["QREQ"]).float()
    Y = torch.from_numpy(d["Y"]).float()
    nb, gap, speed = d["n_beams"], d["gap"], d["speed"]
    train_counts = set(dmeta["splits"]["train"]["counts"])
    train_gaps = set(dmeta["splits"]["train"]["gaps"])
    train_speeds = set(dmeta["splits"]["train"]["speeds"])

    m = DuckTCN()
    m.load_state_dict(torch.load(mdir / "tcn.pt", map_location="cpu"))
    m.eval()
    with torch.no_grad():
        P = m(X, R)
        B = DemandOnly()(X, R)

    def mae(mask, pred):
        mask = torch.from_numpy(mask)
        return float((pred[mask] - Y[mask]).abs().mean() * 1) if mask.any() else float("nan")

    interp = np.isin(nb, list(train_counts))
    ood_count = ~interp
    unseen_gs = np.array([(g not in train_gaps) or (s not in train_speeds)
                          for g, s in zip(gap, speed)]) & interp

    rows = [
        ("interpolation: seen beam counts, unseen height/position", interp),
        ("  of which unseen gap-speed pairs", unseen_gs),
        ("EXTRAPOLATION: 3 beams, a count absent from training", ood_count),
    ]
    print(f"dataset {root.name} hash {dmeta['dataset_hash']}  margin {dmeta['margin_m']} m")
    print(f"model   {mdir.name} {mmeta['n_params']} params\n")
    print(f"{'subset':56s} {'n':>4s} {'TCN MAE':>9s} {'demand':>9s}")
    print("-" * 82)
    out = {}
    for name, mask in rows:
        n = int(mask.sum())
        if not n:
            continue
        out[name.strip()] = {"n": n, "tcn_mae_m": mae(mask, P),
                             "demand_only_mae_m": mae(mask, B)}
        print(f"{name:56s} {n:4d} {mae(mask, P)*1000:8.1f}mm {mae(mask, B)*1000:8.1f}mm")
    print()
    for k in sorted(set(int(v) for v in nb)):
        mask = nb == k
        tag = "trained" if k in train_counts else "UNSEEN COUNT"
        print(f"  {k}-beam ({tag:12s}) n={int(mask.sum()):3d}  "
              f"TCN {mae(mask, P)*1000:6.1f} mm   demand-only {mae(mask, B)*1000:6.1f} mm")
        out[f"{k}_beam"] = {"n": int(mask.sum()), "trained_count": k in train_counts,
                            "tcn_mae_m": mae(mask, P), "demand_only_mae_m": mae(mask, B)}
    (mdir / "experiment_c.json").write_text(json.dumps(
        {"dataset": root.name, "dataset_hash": dmeta["dataset_hash"],
         "model": mdir.name, "train_counts": sorted(train_counts),
         "dip_max_m": DIP_MAX, "subsets": out}, indent=2))
    print(f"\nwrote {mdir}/experiment_c.json")


if __name__ == "__main__":
    main()
