"""Joint native height/forward timing; no generated-motion editing or predictor."""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from scene2motion import scene_relative_duck as duck
from scene2motion import astra_duck as base

DEPTH_LEVELS_M = (.26, .28, .30)
TIMING_LEVELS_S = (-.24, 0., .24)
BASIS_MAX_DERIVATIVE = 192/(25*np.sqrt(5))


@dataclass(frozen=True)
class SpatialWindow:
    descent_start_m: float
    recovery_start_m: float
    source: str

    def validate(self):
        if (not np.isfinite([self.descent_start_m, self.recovery_start_m]).all()
                or self.descent_start_m < 0 or self.descent_start_m+.45 >= self.recovery_start_m
                or self.recovery_start_m+.9 > base.ROUTE_M):
            raise ValueError('invalid spatial crouch window')


def anchor_window(beam, nominal_measurement):
    """Freeze achieved body-envelope events onto the *nominal* command clock.

    Later candidates can change that clock; they must receive their own preview.
    Missing events use an explicit geometry fallback, never an invented arrival.
    """
    nominal = duck.make_spec(beam, duck.DuckCommand())
    times = np.arange(base.FRAMES)/base.FPS
    events = nominal_measurement['overlap_intervals']
    exit_s = nominal_measurement['exit_s']
    if events and exit_s is not None:
        entry_s = events[0]['start_s']
        if not np.isfinite([entry_s, exit_s]).all() or not 0 <= entry_s < exit_s <= times[-1]:
            raise ValueError('invalid observed nominal event clock')
        entry, rear = np.interp([entry_s, exit_s], times, nominal.root_xz[:, 1])
        start, recovery = max(0., entry-.45-.18), rear+.09
        source = 'achieved_envelope_events_on_nominal_command_clock'
        if start+.45 >= recovery or recovery+.9 > base.ROUTE_M:
            source = 'geometry_fallback_out_of_support_events'
            start, recovery = max(0., beam.front_m-1.08), beam.rear_m+1.38
    else:
        source = 'geometry_fallback_censored_events'
        start, recovery = max(0., beam.front_m-1.08), beam.rear_m+1.38
    window = SpatialWindow(float(start), float(recovery), source)
    window.validate()
    return window


def basis(s, start_m, end_m):
    if not np.isfinite([start_m, end_m]).all() or start_m >= end_m:
        raise ValueError('nonempty finite timing support required')
    u = np.clip((np.asarray(s)-start_m)/(end_m-start_m), 0., 1.)
    return 64*u**3*(1-u)**3


def native_arrays(beam, window, depth_m, timing_s):
    window.validate()
    if (not np.isfinite([depth_m, timing_s]).all() or not .26-1e-12 <= depth_m <= .30+1e-12
            or abs(timing_s) > .24+1e-12):
        raise ValueError('joint edit outside declared support')
    spec = duck.make_spec(beam, duck.DuckCommand())
    t = np.arange(base.FRAMES)/base.FPS
    route = spec.root_xz[:, 1]
    lo, hi = float(route[0]), float(route[-1])
    start = max(lo, window.descent_start_m-.45)
    stop = min(hi, window.recovery_start_m+.9+1.2)
    minimum_slope = float(np.min(np.diff(t)/np.diff(route)))
    lower_bound = minimum_slope-abs(timing_s)*BASIS_MAX_DERIVATIVE/(stop-start)
    if not np.isfinite(lower_bound) or lower_bound <= 0:
        raise ValueError('time law is not provably monotone')
    # Include every original knot. Zero timing returns the historical root array exactly.
    grid = np.unique(np.r_[np.linspace(lo, hi, 8193), route, start, stop])
    time_law = np.interp(grid, route, t)+timing_s*basis(grid, start, stop)
    if np.any(np.diff(time_law) <= 0):
        raise ValueError('sampled time law is nonmonotone')
    s = route.copy() if timing_s == 0 else np.interp(t, time_law, grid)
    root = spec.root_xz.copy()
    root[:, 1] = s
    def smooth(x):
        x = np.clip(x, 0., 1.)
        return x**3*(10-15*x+6*x*x)
    weight = smooth((s-window.descent_start_m)/.45)*(1-smooth((s-window.recovery_start_m)/.9))
    from scene2motion.learn.predictor import NOMINAL_PELVIS
    height = NOMINAL_PELVIS-depth_m*weight
    if (not np.array_equal(root[[0, -1]], spec.root_xz[[0, -1]])
            or np.any(np.diff(root[:, 1]) <= 0)):
        raise ValueError('route endpoints or forward ordering changed')
    return root, height, spec.heading.copy(), {
        'depth_m': float(depth_m), 'timing_s': float(timing_s), 'window': asdict(window),
        'basis_support_m': [start, stop], 'time_law_derivative_lower_bound_s_m': lower_bound,
        'native_endpoint_m': [lo, hi], 'last_constraint_time_s': float(t[-1]),
        'requested_speed_min_max_m_s': [float(np.min(np.diff(s)*base.FPS)), float(np.max(np.diff(s)*base.FPS))],
        'postprocessing': 'none; frozen runner decodes directly from generated rotation features'}


def make_spec(beam, window, depth_m, timing_s):
    from scene2motion.constraints import ConstraintSpec
    root, height, heading, info = native_arrays(beam, window, depth_m, timing_s)
    spec = ConstraintSpec(root_xz=root, root_y=height, heading=heading, first_heading=float(heading[0]))
    return spec, info


def request_digest(spec):
    """Hash actual native request arrays, including shape/dtype and masks via fields."""
    h = hashlib.sha256()
    for name in ('root_xz', 'root_y', 'heading'):
        a = np.ascontiguousarray(getattr(spec, name))
        h.update(json.dumps([name, a.dtype.str, list(a.shape)]).encode())
        h.update(a.tobytes())
    h.update(json.dumps({'T': spec.T, 'first_heading': spec.first_heading,
                         'dense_root': spec.root_frames is None}).encode())
    return h.hexdigest()
