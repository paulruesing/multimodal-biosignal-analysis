import pandas as pd
import pytest
from pathlib import Path
import sys

plotly_go = pytest.importorskip("plotly.graph_objects")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.visualizations import plot_category_reassignment_sankey


def _patch_plotly_show(monkeypatch):
    monkeypatch.setattr(plotly_go.Figure, "show", lambda self: None, raising=False)


def test_category_reassignment_sankey_renames_labels_and_uses_raw_color_fallback(monkeypatch):
    _patch_plotly_show(monkeypatch)

    frame = pd.DataFrame(
        {
            "from": ["Happy", "Sad", "Happy"],
            "to": ["Groovy", "Groovy", "Groovy"],
        }
    )
    song_colors = {
        "Happy": "#ff0000",
        "Sad": "#0000ff",
        "Groovy": "#00ff00",
    }

    fig = plot_category_reassignment_sankey(
        category_reassignment_frame=frame,
        song_colors=song_colors,
        preferred_order=["Sad", "Happy"],
        rename_dict={"Happy": "Positive", "Sad": "Negative", "Groovy": "Target"},
        show_title=False,
    )

    assert fig is not None
    annotations = [annotation.text for annotation in fig.layout.annotations]
    assert annotations[:4] == ["Original Category", "Perceived Category", "Negative", "Positive"]
    assert "Happy" not in annotations
    assert "Sad" not in annotations
    assert list(fig.data[0].link.value) == [1, 2]
    assert list(fig.data[0].node.color)[0] == "rgba(0, 0, 255, 0.85)"


def test_category_reassignment_sankey_groups_by_renamed_labels(monkeypatch):
    _patch_plotly_show(monkeypatch)

    frame = pd.DataFrame(
        {
            "from": ["Happy", "Sad", "Happy"],
            "to": ["Classic", "Classic", "Classic"],
        }
    )
    song_colors = {
        "Happy": "#ff0000",
        "Sad": "#0000ff",
        "Classic": "#222222",
    }

    fig = plot_category_reassignment_sankey(
        category_reassignment_frame=frame,
        song_colors=song_colors,
        rename_dict={"Happy": "Enjoyable", "Sad": "Enjoyable", "Classic": "Stable"},
        show_title=False,
    )

    assert fig is not None
    assert list(fig.data[0].link.value) == [3]
    annotations = [annotation.text for annotation in fig.layout.annotations]
    assert "Enjoyable" in annotations
    assert "Stable" in annotations
    assert "Happy" not in annotations
    assert "Sad" not in annotations



