# Findings page

`index.html` is a standalone build of the Scene2Motion findings page — open it with any
browser, no server needed. `artifact.html` is the same content shaped for publishing as a
Claude Artifact (no `<!doctype>` wrapper; the host supplies it).

## Rebuilding

```
$S2M_PY experiments/export_demo_motions.py     # motion payload from committed archives
$S2M_PY docs/site/build.py                     # concatenate parts -> index/artifact
```

The figure numbers in `_payload.js` come from `outputs/figure_data.json`, written by the
analysis scripts; the motion tracks are world-space forward kinematics of archived clips,
not illustrations. Edit `_head.html` (tokens and styles), `_body.html` (prose), or
`_script.html` (players and charts) — never the generated files.

Chart palette is validated, not eyeballed: `#0A8C72`/`#C05A18` light and
`#28A084`/`#CE7238` dark both pass the lightness-band, chroma-floor, CVD-separation,
normal-vision and contrast checks.
