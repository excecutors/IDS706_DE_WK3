import os
import sys
import importlib
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Ensure repo root is importable so `import analysis` finds ../analysis.py
# ──────────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Paths
OUTPUTS_DIR = REPO_ROOT / "outputs"


# ───────────────── Synthetic CSV + Fake S3 (no real AWS) ─────────────────────
def make_synthetic_csv(n_rows: int = 400) -> str:
    start = date(2015, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_rows)]

    rng = np.random.default_rng(42)
    spx = rng.normal(3500, 800, size=n_rows)
    uso = rng.normal(80, 20, size=n_rows).clip(10, 200)
    slv = rng.normal(20, 5, size=n_rows).clip(5, 60)
    eurusd = rng.normal(1.12, 0.05, size=n_rows).clip(0.9, 1.3)

    gld = (
        0.03 * (spx - spx.mean())
        + 0.6 * (slv - slv.mean())
        + 0.2 * (uso - uso.mean())
        - 5.0 * (eurusd - eurusd.mean())
        + 160
        + rng.normal(0, 3, size=n_rows)
    )

    prices = pl.DataFrame(
        {
            "Date": [d.isoformat() for d in dates],
            "SPX": spx,
            "GLD": gld,
            "USO": uso,
            "SLV": slv,
            "EUR/USD": eurusd,
        }
    )
    return prices.write_csv()


class _FakeBody:
    def __init__(self, raw: bytes):
        self._raw = raw

    def read(self) -> bytes:
        return self._raw


class FakeS3Client:
    def __init__(self, csv_text: str):
        self._csv = csv_text

    def get_object(self, Bucket: str, Key: str):
        return {"Body": _FakeBody(self._csv.encode("utf-8"))}


# ─────────────────────────────── Fixtures ─────────────────────────────────────
@pytest.fixture(autouse=True)
def mock_s3(monkeypatch):
    """Mock boto3.client('s3') globally for all tests."""
    csv_text = make_synthetic_csv(n_rows=400)

    def fake_boto3_client(service_name, *args, **kwargs):
        assert service_name == "s3"
        return FakeS3Client(csv_text)

    monkeypatch.setattr("boto3.client", fake_boto3_client, raising=True)


@pytest.fixture
def clean_outputs():
    """Ensure outputs/ exists and is clean (in repo root)."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    for f in ("gld_hist.png", "spx_vs_gld.png"):
        try:
            (OUTPUTS_DIR / f).unlink()
        except FileNotFoundError:
            pass
    return OUTPUTS_DIR


def import_fresh_analysis():
    """Import (or reload) analysis.py from repo root."""
    if "analysis" in sys.modules:
        return importlib.reload(sys.modules["analysis"])
    return importlib.import_module("analysis")


# ─────────────────────────────── Tests ────────────────────────────────────────
def test_end_to_end_runs_and_saves_plots(clean_outputs, capsys):
    mod = import_fresh_analysis()

    out = capsys.readouterr().out
    assert ("R² score:" in out) or ("shape:" in out)

    assert (OUTPUTS_DIR / "gld_hist.png").exists()
    assert (OUTPUTS_DIR / "spx_vs_gld.png").exists()


def test_dataframe_basic_properties():
    mod = import_fresh_analysis()
    assert hasattr(mod, "prices"), "analysis.py must expose `prices`"
    prices = mod.prices
    assert prices.shape[0] > 100

    # Accept either the original 6 columns OR the same 6 + 'Year'
    base = {"Date", "SPX", "GLD", "USO", "SLV", "EUR/USD"}
    cols = set(prices.columns)
    assert base.issubset(cols), f"Missing expected columns. Got: {cols}"

    assert prices["SPX"].dtype == pl.Float64
    assert prices["GLD"].dtype == pl.Float64


def test_filtering_and_grouping():
    mod = import_fresh_analysis()
    assert hasattr(mod, "high_gold")
    assert hasattr(mod, "yearly_avg_gld")

    hg = mod.high_gold
    ya = mod.yearly_avg_gld

    # Robust predicate check in Polars: min(GLD) > 180 when any rows exist
    if hg.shape[0] > 0:
        assert hg["GLD"].min() > 180

    # Yearly aggregation sanity
    assert ya.shape[0] >= 2  # at least a couple of years in synthetic data
    assert {"Year", "avg_GLD"}.issubset(set(ya.columns))



def test_linear_regression_metrics_reasonable():
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error

    mod = import_fresh_analysis()
    X = mod.prices.select(["SPX", "USO", "SLV", "EUR/USD"]).to_numpy()
    y = mod.prices["GLD"].to_numpy()

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = LinearRegression().fit(Xtr, ytr)
    yp = m.predict(Xte)

    r2 = r2_score(yte, yp)
    rmse = np.sqrt(mean_squared_error(yte, yp))

    assert r2 > 0.7
    assert rmse < 20.0


def test_second_import_is_idempotent(clean_outputs):
    _ = import_fresh_analysis()
    # Remove artifacts then reload to ensure they are recreated
    for f in ("gld_hist.png", "spx_vs_gld.png"):
        p = OUTPUTS_DIR / f
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    _ = import_fresh_analysis()
    assert (OUTPUTS_DIR / "gld_hist.png").exists()
    assert (OUTPUTS_DIR / "spx_vs_gld.png").exists()