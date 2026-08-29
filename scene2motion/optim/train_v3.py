"""Train the Phase 3 residual TCN against the optimiser teacher. CPU only."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .model_v3 import DemandOnly, DuckTCN


def load(split: str, root: Path):
    d = np.load(root / f"{split}.npz")
    return (torch.from_numpy(d["X"]).float(), torch.from_numpy(d["QREQ"]).float(),
            torch.from_numpy(d["Y"]).float(), d["n_beams"], d["gap"], d["speed"])


def metrics(pred: torch.Tensor, Y: torch.Tensor, n_beams=None) -> dict:
    e = (pred - Y).detach()
    out = {"mae_m": float(e.abs().mean()),
           "rmse_m": float(torch.sqrt((e ** 2).mean())),
           "peak_mae": float((pred.max(1).values - Y.max(1).values).abs().mean())}
    if n_beams is not None:
        for k in sorted(set(int(v) for v in n_beams)):
            m = torch.from_numpy(np.asarray(n_beams) == k)
            if m.any():
                out[f"mae_{k}beam"] = float((pred[m] - Y[m]).abs().mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="outputs/duck_dataset_v3_m018")
    ap.add_argument("--out", default=None,
                    help="defaults to outputs/duck_model_v3_<dataset tag>")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    root = Path(args.data)
    # A model is only meaningful relative to the dataset that produced it. The previous
    # checkpoint was trained on a margin-0.12 dataset and then read as though it belonged to a
    # 0.18 one, so the binding is now explicit: the model directory is named after the dataset
    # tag and its metadata carries the dataset hash, and training refuses to start without one.
    dmeta = json.loads((root / "meta.json").read_text())
    if "dataset_hash" not in dmeta:
        raise SystemExit(f"{root}/meta.json has no dataset_hash -- it predates the atomic "
                         f"builder and cannot be bound to a model. Rebuild it.")
    out = Path(args.out) if args.out else Path(f"outputs/duck_model_v3_{root.name.split('_')[-1]}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"dataset {root.name}  margin {dmeta['margin_m']} m  hash {dmeta['dataset_hash']}")

    Xtr, Rtr, Ytr, Btr, _, _ = load("train", root)
    Xdv, Rdv, Ydv, Bdv, _, _ = load("dev", root)
    Xte, Rte, Yte, Bte, Gte, Ste = load("test", root)
    model = DuckTCN()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"TCN {model.n_params} params, receptive field {model.receptive_field} samples | "
          f"train {len(Xtr)} dev {len(Xdv)} test {len(Xte)}")

    demand = DemandOnly()
    base = {k: metrics(demand(Xte, Rte), Yte, Bte) for k in ("test",)}["test"]
    print(f"demand-only control: test MAE {base['mae_m']*1000:.1f} mm")

    best, best_state = float("inf"), None
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(Xtr, Rtr), Ytr)
        loss.backward()
        opt.step()
        if (ep + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                dv = metrics(model(Xdv, Rdv), Ydv)
            if dv["mae_m"] < best:
                best = dv["mae_m"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    train_s = time.time() - t0

    with torch.no_grad():
        res = {"arch": "residual_tcn",
               "dataset": root.name, "dataset_hash": dmeta["dataset_hash"],
               "margin_m": dmeta["margin_m"], "dataset_counts": dmeta["counts"],
               "n_params": model.n_params,
               "receptive_field": model.receptive_field, "epochs": args.epochs,
               "seed": args.seed, "train_s": round(train_s, 1),
               "train": metrics(model(Xtr, Rtr), Ytr, Btr),
               "dev": metrics(model(Xdv, Rdv), Ydv, Bdv),
               "test": metrics(model(Xte, Rte), Yte, Bte),
               "demand_only_test": base}
    torch.save(model.state_dict(), out / "tcn.pt")
    (out / "tcn.json").write_text(json.dumps(res, indent=2))
    for k in ("train", "dev", "test"):
        r = res[k]
        extra = "  ".join(f"{kk.replace('mae_','')} {vv*1000:.1f}"
                          for kk, vv in r.items() if kk.startswith("mae_") and "beam" in kk)
        print(f"  {k:5s} MAE {r['mae_m']*1000:5.1f} mm  peak {r['peak_mae']*1000:5.1f} mm"
              + (f"   by beams: {extra}" if extra else ""))
    print(f"  vs demand-only {base['mae_m']*1000:.1f} mm -> "
          f"{100*(1-res['test']['mae_m']/base['mae_m']):.0f} % better")
    print(f"  trained {train_s:.1f}s on CPU -> {out/'tcn.pt'}")


if __name__ == "__main__":
    main()
