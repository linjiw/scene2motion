import importlib.util
from pathlib import Path
import sys
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs/site"
sys.path.insert(0, str(SITE))
import build_progress as progress
from export_publication import allowed


def test_public_snapshot_matches_all_assigned_receipts():
    data = progress.snapshot()
    assert data["executions"] == 576 and len(data["cells"]) == 9
    assert data["cells"][6]["local_passes"] == data["cells"][6]["local_and_recovery"] == 17
    assert data["physical_contact"] is None
    assert data["high_quality_motion_qualified"] is None
    assert data["sensitivity"][5]["fixed3"] == 17


def test_generated_progress_is_current_and_shared():
    data = progress.snapshot()
    html = progress.render(data).rstrip()
    for path in (ROOT / "docs/index.html", SITE / "_body.html", SITE / "index.html", SITE / "artifact.html"):
        text = path.read_text()
        block = text.split(progress.BEGIN)[1].split(progress.END)[0].strip()
        assert block == html.strip()
        assert "No obstacle traversal has been completed" not in text


@pytest.mark.parametrize("text", ["none", progress.BEGIN, progress.END + progress.BEGIN, progress.BEGIN * 2 + progress.END])
def test_invalid_markers_refused(text):
    with pytest.raises(ValueError):
        progress.replace_block(text, "new")


def test_rebuild_is_idempotent():
    text = "before" + progress.BEGIN + "old" + progress.END + "after"
    result = progress.replace_block(text, "new")
    assert progress.replace_block(result, "new") == result
    assert result.startswith("before") and result.endswith("after")


@pytest.mark.parametrize("path", ["outputs/model.pt", "outputs/motion.pkl", "docs/paper-draft-private.md", "docs/REPORT.md", "../docs/index.html", "/docs/index.html", "outputs/run/sonic.log"])
def test_export_excludes_unreleased_and_unrelated_files(path):
    assert not allowed(path)


def test_publication_allows_only_selected_evidence():
    assert allowed("docs/index.html")
    assert allowed("docs/media/s4434_reference_repair.mp4")
    assert allowed("docs/assets/staged.mp4")
    assert allowed(progress.SOURCES["a5"])
    assert not allowed("outputs/astra_a5_encoder_d0/fixed3.pkl")


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls, self.ids = [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for key in ("href", "src", "poster"):
            if key in attrs:
                self.urls.append(attrs[key])
        if "id" in attrs:
            self.ids.append(attrs["id"])


@pytest.mark.parametrize("page", ["docs/index.html", "docs/site/index.html"])
def test_static_assets_and_internal_anchors_exist(page):
    path = ROOT / page
    parsed = Links()
    parsed.feed(path.read_text())
    assert len(parsed.ids) == len(set(parsed.ids))
    for url in parsed.urls:
        value = urlsplit(url)
        if value.scheme or value.netloc:
            continue
        target = path.parent / unquote(value.path) if value.path else path
        assert target.exists(), (page, url)
        assert allowed(target.resolve().relative_to(ROOT).as_posix()), ("missing from publication allowlist", page, url)
        if not value.path and value.fragment:
            assert value.fragment in parsed.ids, (page, url)


def test_recent_progress_keeps_partial_gate_separate_from_pilot_rate():
    data = progress.snapshot()['recent']
    assert data['a16']['arms']['no_phase']['selected_passes'] == 5
    assert data['a16']['arms']['fixed_shallow']['selected_passes'] == 4
    assert data['a17']['arms'] is None
    assert data['a17']['full_cohort_executed'] is False
    assert data['a17']['development_lead_gate']['selected_net_gain_over36'] == 0
    assert data['a17']['new_unique_scientific_previews'] == 2
    assert data['a17']['new_context_workload_executions'] == 62
    assert data['a17']['unexecuted_planned_jobs'] == 35


@pytest.mark.parametrize('mutation', ['gate', 'absolute_rates', 'tier'])
def test_recent_narrative_refuses_changed_evidence(mutation):
    import json
    data = {name: json.loads((ROOT/path).read_text()) for name, path in progress.SOURCES.items()}
    if mutation == 'gate':
        data['a17']['development_lead_gate']['pass'] = True
    elif mutation == 'absolute_rates':
        data['a17']['arms'] = {'no_phase': {'assigned': 36, 'passes': 0}}
    else:
        data['a17']['present_evaluations'] = 64
    with pytest.raises(ValueError):
        progress.recent_snapshot(data, ROOT)
