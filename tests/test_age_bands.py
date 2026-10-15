import pandas as pd
import pytest

from src.cleaning import age_band, AGE_BANDS, add_age_band


def test_age_band_boundaries():
    assert age_band(17) == "Under 18"
    assert age_band(18) == "18-25"
    assert age_band(25) == "18-25"
    assert age_band(26) == "26-35"
    assert age_band(35) == "26-35"
    assert age_band(36) == "36-45"
    assert age_band(45) == "36-45"
    assert age_band(46) == "46 and above"
    assert age_band(100) == "46 and above"
    assert age_band(None) == "Unknown / missing / invalid age"
    assert age_band("abc") == "Unknown / missing / invalid age"
    assert age_band(-5) == "Unknown / missing / invalid age"
    assert age_band(150) == "Unknown / missing / invalid age"


def test_age_bands_no_overlap():
    # boundaries contiguous
    assert AGE_BANDS == ["Under 18", "18-25", "26-35", "36-45", "46 and above", "Unknown / missing / invalid age"]


def test_add_age_band():
    df = pd.DataFrame({"Age": [15, 20, 30, 40, 50, None]})
    out = add_age_band(df)
    assert "AgeBand" in out.columns
    assert list(out["AgeBand"].astype(str)) == ["Under 18", "18-25", "26-35", "36-45", "46 and above", "Unknown / missing / invalid age"]
