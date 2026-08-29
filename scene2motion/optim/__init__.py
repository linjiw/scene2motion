"""Phase 3: an optimised, composable body scheduler.

Phase 2 learned to imitate the heuristic and therefore inherited its ceiling. Phase 3 replaces
the teacher: a schedule optimiser that minimises crouch effort and jerk subject to a clearance
constraint, evaluated through a fitted model of how ARDY actually responds to a duck command.
The learned network then distils the OPTIMISER, so it can do things the rule never did --
crouch only as deep as the beam requires, and decide for itself whether two nearby beams share
one crouch or get two.
"""
