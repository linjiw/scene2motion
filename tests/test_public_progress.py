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
        if not value.path and value.fragment:
            assert value.fragment in parsed.ids, (page, url)
