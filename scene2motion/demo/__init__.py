"""Thin demo layer: scene -> route -> body adaptation -> ARDY motion -> visualisation.

Deliberately thin. Everything load-bearing already exists in `scene2motion` and is reused
as-is: `planner.plan` for routes, `program`/`planner.plan_to_spec` for the body request,
`runner.ArdyRunner` for generation, `robot.G1Body` for collision geometry. This package adds
a parametric scene, a preference->planner-argument mapping, a content-addressed clip cache,
a dependency-light HTTP UI, and nothing else.
"""
