# Partial-beam interactive demo

`scene + start + goal + preference -> route -> body adaptation -> frozen ARDY -> G1`

One overhead beam covering part of a corridor. Under it is shorter; around it keeps the robot
upright. The user picks the preference, the planner picks the route and the body programme, and
the frozen prior generates the motion.

## Run

```
cd /home/linjiw/scene2motion
/home/linjiw/ardy/.venv/bin/python -m scene2motion.demo.app --port 8000
# -> http://127.0.0.1:8000
```

Planning is live on every slider move (~130 ms for all three preferences, CPU only).
**Generate motion** is separate and cached, so the UI never blocks on the GPU and a fully
cached session never loads the model.

Deep links restore a state and can auto-generate — useful for screenshots and recordings:

```
/?height=1.00&width=1.45&preference=shortest&auto=1&frame=58
```

## Seed the cache

```
python -m scene2motion.demo.seed_cache --heights 1.60 1.30 1.15 1.00 0.90 \
                                       --preferences shortest upright clearance
```
15 clips take ~10 s. Entries are content-addressed and independent, so this can be stopped and
resumed at any point.

## Check it still works

```
python -m pytest tests/test_demo.py -q          # 14 CPU-only tests, no ARDY
python -m scene2motion.demo.acceptance          # 16 checks against the live HTTP API
```

## V2: body planners and the schedule plot

The **body layer** selector chooses who fills the duck channel. The route is identical in every
case -- that is the route/body split, so any difference in the clip is attributable to the body
layer alone.

| layer | what it is |
|---|---|
| Heuristic | the calibrated mode lattice, dilated and smoothed by `plan_to_spec` |
| Phase-2 Learned | a 3 185-param CNN trained to imitate the heuristic |
| Phase-3 Optimized | a 14 k-param residual TCN distilling the convex-QP schedule optimiser |

The **duck-schedule plot** under the animation shows all of them on one route-distance axis,
with the beam spans shaded, the heuristic dashed, the QP teacher dotted and the selected layer
solid. The plotted schedule is the one that generated the clip -- both come from
`demo/schedules.py`, which exists so they cannot drift apart.

The **scene preset** switches between one beam and two, with a gap slider. Two stable examples:

* `?n_beams=2&gap=2.0` -- one continuous crouch across both beams;
* `?n_beams=2&gap=5.0` -- duck, stand back up, duck again.

Which of those happens is not a rule. The optimiser decides it by minimising crouch effort and
jerk against a lag of 0.19 s, and the boundary lands where standing back up costs less than
staying down.

## Clearance margin

`MARGIN_M` in `optim/scheduler.py` is 0.18 m and is not a round number. `optim/verify_margin.py`
sweeps it against the real prior: 0.12 m looked generous beside a 43 mm surrogate holdout error
and produced actual collisions (80 % collision-free, worst clearance -18 mm). 0.18 m is the
smallest swept value restoring 100 % collision-free, at 24.3 cm of mean peak crouch. The margin
has to cover the surrogate's error *and* ARDY's own 30-74 mm per-seed scatter.

Model checkpoints are versioned by the margin they were trained for (`duck_model_v3_m018`), and
`demo/schedules.py` refuses to serve a checkpoint whose recorded margin differs from the one the
optimiser is solving with.

## What the three preferences are

They are three argument sets to the *same* `planner.plan`, not three planners:

| preference | mechanism | result at a 1.00 m beam |
|---|---|---|
| Shortest Path | body-mode costs discounted (`SHORTEST_COST_ALPHA`) so route length dominates | 8.00 m, under, `duck_max` |
| Stay Upright | `allow_modes` excludes every duck mode | 8.50 m, around |
| Maximum Clearance | as above, plus the beam footprint dilated by 0.55 m via `forbid_boxes` | 8.62 m, around, wider berth |

The discount matters: under the shipped costs, holding a deep duck for ~1.5 m outweighs a 0.5 m
detour, so the default planner walks *around* a beam it could duck under. That is right for its
objective and wrong for a control labelled "Shortest Path".

## What the demo is allowed to claim

Three states are kept explicitly distinct and never collapsed:

- **kinematic collision-free** — the generated qpos clears the scene under MuJoCo with the
  measured 40 mm `BODY_MARGIN`. This is what the badge reports.
- **SONIC tracked** — a physics controller reproduced it. The demo does **not** run this inline
  and always reports `no`.
- **not physics validated** — the default. Shown alongside the collision badge every time.

The animation draws the robot's own collision primitives rather than a mesh, so what is on
screen is exactly the geometry the collision status refers to.

## Files

| file | role |
|---|---|
| `scene_builder.py` | the parametric partial beam; only height and width move |
| `strategy_planner.py` | preference -> `plan()` arguments; extracts only defensible facts |
| `cache.py` | content-addressed clips with provenance |
| `ardy_runner.py` | lazy model load, 5 denoising steps, collision check on generate |
| `renderer.py` | qpos -> compact side/BEV drawing payload |
| `app.py` | stdlib HTTP server + single-page UI |
| `seed_cache.py` | populate the cache |
| `acceptance.py` | V1 acceptance criteria against the live API |

## Known limits

- One scene family. The beam is the only obstacle and the corridor is fixed.
- At a shallow beam (~1.30 m) the planner holds a light duck for the whole path rather than
  localising it — cheaper than paying `MODE_SWITCH_COST` twice. The panel says so
  (`duck window: held throughout`) instead of implying a crouch that did not happen.
- Step-over is deliberately absent: EXP-014 puts a correctly-gated one at 0.375 tracked success
  and only at 0.08 m amplitude, which is too weak to present as a capability.
