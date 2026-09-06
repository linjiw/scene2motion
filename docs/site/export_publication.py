"""Export a static, allowlisted GitHub Pages tree from a committed source revision.

No pushes or Pages-setting changes occur here. Never export motion arrays, pickle
payloads, checkpoints, process logs, private drafts or the whole research worktree.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from build_progress import SOURCES

ROOT = Path(__file__).resolve().parents[2]
EXTRA = {
    "astra.md", "docs/.nojekyll", "docs/index.html", "docs/progress.json",
    "docs/ramp-exp031-constructive-step-repair-result-2026-09-04.md",
    "docs/ramp-exp024-reference-contract-result-2026-09-04.md",
    "docs/execution-aware-step-compiler-v2-plan-2026-09-04.md",
    "scene2motion/recovery_sensitivity.py", "experiments/analyze_astra_a5_progress_sensitivity.py",
    "tests/test_recovery_sensitivity.py", "tests/test_public_progress.py",
}


def allowed(path):
    p = Path(path)
    if p.is_absolute() or ".." in p.parts:
        return False
    return (path in EXTRA or path in SOURCES.values()
        or (p.parent.as_posix() == "docs" and p.name.startswith("astra-") and p.suffix == ".md")
        or (p.parent.as_posix() == "docs/site" and p.suffix in {".html", ".css", ".js", ".py", ".md"})
        or (p.parent.as_posix() in {"docs/figures", "docs/media"} and p.suffix in {".svg", ".png", ".jpg", ".mp4", ".json", ".pdf"}))


def export(destination, root=ROOT):
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("publication output must be empty")
    if destination == root or root in destination.parents:
        raise ValueError("publication output must be outside the research workspace")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    tracked = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit], cwd=root, text=True).splitlines()
    chosen = sorted(p for p in tracked if allowed(p))
    required = EXTRA | set(SOURCES.values())
    if not required.issubset(chosen):
        raise ValueError("required publication files are not committed")
    hashes = {}
    for path in chosen:
        content = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=root)
        if Path(path).suffix in {".html", ".md"}:
            text = content.decode()
            for target in chosen:
                text = text.replace("https://github.com/linjiw/scene2motion/blob/master/" + target,
                                    "https://github.com/linjiw/scene2motion/blob/gh-pages/" + target)
            content = text.encode()
        dest = destination / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        hashes[path] = hashlib.sha256(content).hexdigest()
    manifest = {"schema_version": "scene2motion-pages-export-v1", "research_source_commit": commit,
        "files": hashes, "motion_payloads_included": False,
        "scope": "static pages, selected result/protocol documents and JSON receipts; not a motion dataset release"}
    (destination / "docs/publication.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Exported {len(chosen)} committed, allowlisted files from {commit}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    export(parser.parse_args().out)
