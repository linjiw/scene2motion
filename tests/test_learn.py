"""Tests for the learned duck layer. CPU only; the trained checkpoint must exist."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion.demo.strategy_planner import SHORTEST_MODES
from scene2motion.learn.dataset import NO_BEAM_HEIGHT, SPLITS, build_scene, duck_label
from scene2motion.learn.route_profile import CHANNELS, N_SAMPLES, profile
from scene2motion.planner import plan

CKPT = Path("outputs/duck_model/cnn.pt")
needs_ckpt = pytest.mark.skipif(not CKPT.exists(), reason="model not trained")


def route(h, x=4.0, w=1.75, **kw):
    sc = build_scene(h, x, w, **kw)
    p = plan(sc, "adaptive", modes_override=SHORTEST_MODES)
    return sc, p


# ---- splits --------------------------------------------------------------------------

def test_split_geometry_is_disjoint():
    """A test score must be about unseen geometry, not unseen noise."""
    tr, dv, te = (set(SPLITS[k]["heights"]) for k in ("train", "dev", "test"))
    assert tr & dv == set() and tr & te == set() and dv & te == set()
    ptr, pte = set(SPLITS["train"]["positions"]), set(SPLITS["test"]["positions"])
    assert ptr & pte == set()


# ---- profile -------------------------------------------------------------------------

def test_profile_shape_and_units():
    sc, p = route(1.00)
    P = profile(sc, p.xy)
    assert P.shape == (N_SAMPLES, len(CHANNELS))
    assert np.isfinite(P).all()


def test_overhead_channel_sees_the_beam():
    sc, p = route(1.00)
    P = profile(sc, p.xy)
    assert P[:, 0].min() == pytest.approx(1.00, abs=0.02)


def test_time_to_restriction_counts_down_to_the_beam():
    sc, p = route(1.00)
    t = profile(sc, p.xy)[:, CHANNELS.index("time_to_restriction")]
    lo = int(np.argmin(t))
    assert t[lo] == pytest.approx(0.0, abs=1e-6)
    assert t[max(0, lo - 6)] > t[max(0, lo - 2)], "must decrease on approach"


def test_no_beam_leaves_the_overhead_channel_clear():
    sc, p = route(NO_BEAM_HEIGHT)
    assert profile(sc, p.xy)[:, 0].min() > 2.0


# ---- labels --------------------------------------------------------------------------

def test_label_is_zero_without_a_beam():
    sc, p = route(NO_BEAM_HEIGHT)
    assert duck_label(p).max() == pytest.approx(0.0, abs=1e-6)


def test_label_is_deeper_for_a_lower_beam():
    peaks = [duck_label(route(h)[1]).max() for h in (1.20, 1.10, 1.00)]
    assert peaks[0] <= peaks[1] <= peaks[2]


def test_label_anticipates_the_beam():
    """The duck must begin before the robot is under the beam, not while it is."""
    sc, p = route(1.00, x=4.0)
    y = duck_label(p)
    xy = np.asarray(p.xy)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    total = seg.sum()
    onset_s = np.where(y > 0.03)[0][0] / (len(y) - 1) * total
    assert onset_s < 4.0, f"duck starts at {onset_s:.2f} m, beam is at 4.0 m"


# ---- model ---------------------------------------------------------------------------

@needs_ckpt
def test_model_is_small():
    from scene2motion.learn.model import DuckCNN
    assert DuckCNN().n_params < 6000


@needs_ckpt
def test_prediction_is_nonnegative_and_bounded():
    from scene2motion.learn.predictor import predict_dip
    sc, p = route(1.00)
    d = predict_dip(sc, p.xy)
    assert d.shape == (N_SAMPLES,)
    assert (d >= 0).all() and d.max() <= 0.55


@needs_ckpt
def test_model_predicts_no_duck_without_a_beam():
    from scene2motion.learn.predictor import predict_dip
    sc, p = route(NO_BEAM_HEIGHT)
    assert predict_dip(sc, p.xy).max() < 0.03


@needs_ckpt
def test_model_is_monotone_in_beam_height():
    from scene2motion.learn.predictor import predict_dip
    peaks = [predict_dip(*route(h)[:1], route(h)[1].xy).max() for h in (1.20, 1.10, 1.00)]
    assert peaks[0] <= peaks[1] <= peaks[2], peaks


@needs_ckpt
def test_model_anticipates_and_the_onset_moves_with_the_beam():
    from scene2motion.learn.predictor import predict_dip
    onsets = []
    for x in (3.0, 4.0, 5.0):
        sc, p = route(1.00, x=x)
        d = predict_dip(sc, p.xy)
        idx = np.where(d > 0.03)[0]
        assert len(idx), f"no duck predicted for a beam at {x} m"
        xy = np.asarray(p.xy)
        total = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
        onsets.append(idx[0] / (len(d) - 1) * total)
    assert onsets[0] < onsets[1] < onsets[2], onsets


@needs_ckpt
def test_spec_from_dip_writes_only_the_duck_channel():
    from scene2motion.learn.predictor import predict_dip, spec_from_dip
    sc, p = route(1.00)
    spec = spec_from_dip(sc, p.xy, predict_dip(sc, p.xy), fps=25.0)
    assert spec.root_y is not None and spec.pos_frames is None
    assert spec.root_y.min() < 0.78, "a duck must lower the pelvis"


# ---- provenance ----------------------------------------------------------------------
# Independent verification found two prose/artifact mismatches: a parameter count quoted as
# 3137 when the model has 3185, and a two-beam ladder described with five gaps while the
# committed probe ran three. Both were drift between what was run and what was written. These
# assert the artifacts describe the code that produced them, which is the part a reader relies
# on and the part that silently rots.

ARTIFACTS = Path("outputs/duck_model")
has_artifacts = pytest.mark.skipif(not (ARTIFACTS / "probes.json").exists(),
                                   reason="probes not run")


@needs_ckpt
def test_committed_model_json_matches_the_live_model():
    import json
    from scene2motion.learn.model import DuckCNN, DuckMLP
    for arch, cls in (("cnn", DuckCNN), ("mlp", DuckMLP)):
        f = ARTIFACTS / f"{arch}.json"
        if not f.exists():
            continue
        assert json.loads(f.read_text())["n_params"] == cls().n_params, (
            f"{arch}.json records a parameter count the model no longer has")


@has_artifacts
def test_probes_artifact_matches_the_probe_configuration():
    import json
    from scene2motion.learn.probes import TWO_BEAM_GAPS
    pr = json.loads((ARTIFACTS / "probes.json").read_text())
    gaps = tuple(r["gap_m"] for r in pr["two_beams"])
    assert gaps == TWO_BEAM_GAPS, (
        f"probes.json ran gaps {gaps}, the probe now defines {TWO_BEAM_GAPS}")
    assert "excess_crouch" in pr, "excess-crouch claim must be backed by the artifact"
    assert len(pr["excess_crouch"]) >= 6


@has_artifacts
def test_latency_is_reported_as_a_median_with_a_range():
    import json
    lat = json.loads((ARTIFACTS / "probes.json").read_text())["latency"]
    for k in ("repeats", "heuristic_ms_range", "learned_ms_range"):
        assert k in lat, f"latency artifact missing {k}"
    lo, hi = lat["heuristic_ms_range"]
    assert lo <= lat["heuristic_ms"] <= hi
