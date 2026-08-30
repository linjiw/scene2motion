"""Phase 4: generate -> verify -> repair -> select.

The Phase 3 scheduler plans against a fitted surrogate whose holdout error is ~43 mm, against a
prior whose own per-seed scatter is 30-74 mm. A 0.18 m margin covers that on the scenes it was
swept over. This package closes the loop instead of trusting it: generate, measure the actual
clearance the real motion leaves, and repair locally where it falls short.
"""
