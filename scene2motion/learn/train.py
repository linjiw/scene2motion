"""Train the duck-schedule model. CPU only -- LUCID owns the GPU.

Model selection is on the DEV split, which is disjoint in beam height and position from both
train and test, so the checkpoint that gets evaluated was never chosen using test geometry.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .model import MODELS


def load(split: str, root: Path) -> tuple[torch.Tensor, torch.Tensor]:
    d = np.load(root / f"{split}.npz")
    return torch.from_numpy(d["X"]).float(), torch.from_numpy(d["Y"]).float()


def evaluate(model, X, Y) -> dict:
    model.eval()
    with torch.no_grad():
        P = model(X)
    err = (P - Y)
    peak_p, peak_y = P.max(1).values, Y.max(1).values
    ducks_y, ducks_p = peak_y > 0.02, peak_p > 0.02
    return {
        "mae_m": float(err.abs().mean()),
        "rmse_m": float(torch.sqrt((err ** 2).mean())),
        "peak_mae_m": float((peak_p - peak_y).abs().mean()),
        "duck_detect_acc": float((ducks_p == ducks_y).float().mean()),
        "false_duck_rate": float((ducks_p & ~ducks_y).float().mean()),
        "missed_duck_rate": float((~ducks_p & ducks_y).float().mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="outputs/duck_dataset")
    ap.add_argument("--out", default="outputs/duck_model")
    ap.add_argument("--arch", default="cnn", choices=sorted(MODELS))
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    Xtr, Ytr = load("train", root)
    Xdv, Ydv = load("dev", root)
    Xte, Yte = load("test", root)
    model = MODELS[args.arch]()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"{args.arch}: {model.n_params} params | train {len(Xtr)} dev {len(Xdv)} "
          f"test {len(Xte)}")

    best, best_state, hist = float("inf"), None, []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(Xtr), Ytr)
        loss.backward()
        opt.step()
        if (ep + 1) % 25 == 0:
            dv = evaluate(model, Xdv, Ydv)
            hist.append({"epoch": ep + 1, "train_mse": float(loss), **dv})
            if dv["mae_m"] < best:
                best = dv["mae_m"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    train_s = time.time() - t0

    res = {"arch": args.arch, "n_params": model.n_params, "epochs": args.epochs,
           "seed": args.seed, "train_s": round(train_s, 1),
           "train": evaluate(model, Xtr, Ytr), "dev": evaluate(model, Xdv, Ydv),
           "test": evaluate(model, Xte, Yte), "history": hist[-8:]}
    torch.save(model.state_dict(), out / f"{args.arch}.pt")
    (out / f"{args.arch}.json").write_text(json.dumps(res, indent=2))
    for k in ("train", "dev", "test"):
        r = res[k]
        print(f"  {k:5s} mae {r['mae_m']*1000:5.1f} mm | peak mae {r['peak_mae_m']*1000:5.1f} mm"
              f" | duck-detect {r['duck_detect_acc']:.3f}"
              f" (false {r['false_duck_rate']:.3f} missed {r['missed_duck_rate']:.3f})")
    print(f"  trained in {train_s:.1f}s on CPU -> {out / (args.arch + '.pt')}")


if __name__ == "__main__":
    main()
