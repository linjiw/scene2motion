"""Phase 2: the learned middle layer.

`route-local scene profile -> duck schedule`. The route stays classical (A* is saturated on
this problem); only the body adaptation along a GIVEN route is learned. Labels come from the
existing heuristic planner, so this is a distillation of a working system into something that
consumes continuous geometry rather than a discrete mode lattice.
"""
