# tests/test_data_loader.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import sys
from pathlib import Path

# Add the src directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import src.data_loader as dl


def test_setup_directories_creates_data_raw(tmp_path, monkeypatch):
    """
    setup_directories() builds the path:
      script_dir = Path(__file__).resolve().parent
      project_root = script_dir.parent
      raw_data_dir = project_root / "data" / "raw"
    So we monkeypatch dl.__file__ to simulate a repo like:
      tmp_path/
        src/data_loader.py
    """
    fake_src_dir = tmp_path / "src"
    fake_src_dir.mkdir(parents=True, exist_ok=True)
    fake_file = fake_src_dir / "data_loader.py"
    fake_file.write_text("# dummy file for tests")

    # Make the module believe its __file__ is inside tmp_path/src/
    monkeypatch.setattr(dl, "__file__", str(fake_file))

    raw_dir = dl.setup_directories()

    assert isinstance(raw_dir, Path)
    assert raw_dir.exists()
    assert raw_dir.is_dir()

    # Expected: tmp_path/data/raw
    assert raw_dir == tmp_path / "data" / "raw"


def test_download_yahoo_data_returns_false_on_empty(monkeypatch, tmp_path):
    # Mock yf.download to return an empty DataFrame
    def fake_download(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(dl.yf, "download", fake_download)

    ok = dl.download_yahoo_data(
        ticker="USDCHF=X",
        output_filename="usd_chf.csv",
        data_dir=tmp_path,
        start_date="2020-01-01",
        end_date="2020-12-31",
    )

    assert ok is False
    assert not (tmp_path / "usd_chf.csv").exists()


def test_download_yahoo_data_saves_csv_on_success(monkeypatch, tmp_path):
    # Mock yf.download to return a non-empty DataFrame
    df = pd.DataFrame(
        {
            "Open": [1.0, 1.1],
            "High": [1.2, 1.3],
            "Low": [0.9, 1.0],
            "Close": [1.05, 1.15],
            "Adj Close": [1.05, 1.15],
            "Volume": [100, 200],
        },
        index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
    )

    def fake_download(*args, **kwargs):
        return df

    monkeypatch.setattr(dl.yf, "download", fake_download)

    filename = "usd_chf.csv"
    ok = dl.download_yahoo_data(
        ticker="USDCHF=X",
        output_filename=filename,
        data_dir=tmp_path,
        start_date="2020-01-01",
        end_date="2020-12-31",
    )

    assert ok is True
    out = tmp_path / filename
    assert out.exists()

    # Basic check: CSV readable + not empty
    loaded = pd.read_csv(out)
    assert len(loaded) == 2
