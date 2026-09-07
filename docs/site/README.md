# Findings page

`index.html` is a standalone build of the Scene2Motion findings page — open it with any
browser, no server needed. `artifact.html` is the same content shaped for publishing as a
Claude Artifact (no `<!doctype>` wrapper; the host supplies it).

## Rebuilding

The current-progress section is shared by both pages. Edit `_progress.html` for prose
and `progress.css` for its layout; `build_progress.py` validates selected A5 and A10–A17/audit
receipts, generates `docs/progress.json`, and replaces only the marked blocks in
`docs/index.html` and `_body.html`. `build.py` runs this step automatically. Do not
manually edit the generated tables or present old stepping videos as recent duck footage.
The A17 two-case figure is hash-checked and embedded for offline viewing.
No simulator, model loading or motion payload is needed for this progress build.

```
$S2M_PY experiments/export_demo_motions.py           # skeleton tracks from archives
MUJOCO_GL=glfw $S2M_PY experiments/render_demo_videos.py   # MuJoCo simulation video
MUJOCO_GL=egl $S2M_PY experiments/render_step_repair_demo.py --out docs/media
$S2M_PY docs/site/make_payload.py                    # tracks + videos + chart numbers -> _payload.js
$S2M_PY docs/site/build.py                           # concatenate -> index/artifact
```

Only the last two steps are needed when the prose or a chart number changes; the first two
touch the archives and the renderer.

Rendering needs `ffmpeg` and a working MuJoCo OpenGL backend. The original five inline clips
were built with `MUJOCO_GL=glfw`; the standalone EXP-031 reference-repair comparison is
headless-safe with `MUJOCO_GL=egl` on this host. The five inline clips encode to ~0.55 MB total,
small enough to inline as data URIs; the artifact limit is 16 MB and the built page is approximately
1.1 MB. The two externally linked evidence videos live under `docs/media/` with manifests beside
them.

The two charts are read by `make_payload.py` straight out of the committed analysis
ledgers — the event-time histogram from `outputs/analysis_event_frames/receipt.json`, the
exact-position clearance curve from `outputs/analysis_exact_centre_cost_curve/curve.jsonl`
— and written to `outputs/figure_data.json` with the sha256 of each input beside them, so
no chart number is ever retyped. The motion tracks are world-space forward kinematics of
archived clips, not illustrations. Edit `_head.html` (tokens and styles), `_body.html`
(prose), or `_script.html` (players and charts) — never the generated files.

This page is the companion to the project page at `docs/index.html`, which carries the
committed figures under `docs/figures/`. Both must agree; when a result lands, update both.

## Publishing without releasing motion payloads

GitHub Pages uses the dedicated `gh-pages` branch, `/docs` directory. The research
branch may contain unpublished payloads or drafts; do not push it merely to deploy
the site. After rebuilding, testing and committing the intended source changes:

```bash
python3 docs/site/export_publication.py --out /absolute/empty/publication-directory
```

The exporter reads the committed revision, not dirty workspace files, and writes an
allowlisted tree with `docs/publication.json` recording file hashes and source commit.
It includes built pages, historical visual assets, selected result/protocol documents
and JSON receipts. It excludes pickle/array/checkpoint payloads, process logs, private
paper drafts and the unrelated research ledger. Existing master links are redirected
only when the target is included in the publication tree. It performs no network writes.

Commit that export on `gh-pages` and use a normal fast-forward push; never force-push
or copy the whole research repository. Verify the Pages build's commit and fetch the
live `progress.json` and `publication.json` before claiming the website is updated.

Chart palette is validated, not eyeballed: `#0A8C72`/`#C05A18` light and
`#28A084`/`#CE7238` dark both pass the lightness-band, chroma-floor, CVD-separation,
normal-vision and contrast checks.
