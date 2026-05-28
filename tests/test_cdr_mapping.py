"""Tests for sterling/cdr_mapping.py — CDR catalog completeness and helpers."""

import pytest

from sterling import cdr_mapping


class TestCDRUniverse:
    def test_all_keys_end_with_ne(self):
        for ticker in cdr_mapping.CDR_UNIVERSE:
            assert ticker.endswith(".NE"), f"{ticker} does not end with .NE"

    def test_all_underlyings_are_plain_us_symbols(self):
        """Underlying US symbols should not contain exchange suffixes."""
        for ne_ticker, underlying in cdr_mapping.CDR_UNIVERSE.items():
            assert "." not in underlying, (
                f"{ne_ticker} underlying {underlying!r} contains a dot — "
                "should be a plain US symbol like AMZN, not AMZN.US"
            )

    def test_no_duplicate_ne_tickers(self):
        tickers = list(cdr_mapping.CDR_UNIVERSE.keys())
        assert len(tickers) == len(set(tickers))

    def test_no_duplicate_underlyings(self):
        underlyings = list(cdr_mapping.CDR_UNIVERSE.values())
        assert len(underlyings) == len(set(underlyings)), (
            "Two CDR tickers map to the same underlying — check for duplicates"
        )

    def test_minimum_catalog_size(self):
        """Sanity: catalog should have at least 50 tickers."""
        assert len(cdr_mapping.CDR_UNIVERSE) >= 50

    def test_known_tickers_present(self):
        """Key liquid CDRs must be in the catalog."""
        required = ["AMZN.NE", "AAPL.NE", "MSFT.NE", "NVDA.NE", "META.NE",
                    "JPM.NE", "V.NE", "UNH.NE", "JNJ.NE", "XOM.NE"]
        for t in required:
            assert t in cdr_mapping.CDR_UNIVERSE, f"{t} missing from CDR_UNIVERSE"

    def test_cdr_tickers_list_matches_universe_keys(self):
        assert set(cdr_mapping.CDR_TICKERS) == set(cdr_mapping.CDR_UNIVERSE.keys())


class TestIsCDR:
    def test_ne_ticker_is_cdr(self):
        assert cdr_mapping.is_cdr("AMZN.NE") is True

    def test_to_ticker_not_cdr(self):
        assert cdr_mapping.is_cdr("SHOP.TO") is False

    def test_plain_us_ticker_not_cdr(self):
        assert cdr_mapping.is_cdr("AMZN") is False

    def test_tsxv_not_cdr(self):
        assert cdr_mapping.is_cdr("ABX.V") is False

    @pytest.mark.parametrize("ticker", ["AAPL.NE", "MSFT.NE", "NVDA.NE"])
    def test_parametrized_cdr(self, ticker):
        assert cdr_mapping.is_cdr(ticker) is True


class TestGetUnderlying:
    def test_amzn_ne_returns_amzn(self):
        assert cdr_mapping.get_underlying("AMZN.NE") == "AMZN"

    def test_v_ne_returns_v(self):
        assert cdr_mapping.get_underlying("V.NE") == "V"

    def test_non_cdr_returns_unchanged(self):
        assert cdr_mapping.get_underlying("SHOP.TO") == "SHOP.TO"
        assert cdr_mapping.get_underlying("RY.TO") == "RY.TO"

    def test_unknown_ne_returns_unchanged(self):
        """A .NE ticker not in the catalog is returned as-is (safe fallback)."""
        assert cdr_mapping.get_underlying("FAKE.NE") == "FAKE.NE"

    def test_all_catalog_tickers_resolve(self):
        for ne_ticker, expected_underlying in cdr_mapping.CDR_UNIVERSE.items():
            assert cdr_mapping.get_underlying(ne_ticker) == expected_underlying


class TestVolumeMin:
    def test_cdr_volume_min_higher_than_tsx_floor(self):
        from sterling.analyst import MIN_AVG_VOLUME
        assert cdr_mapping.CDR_VOLUME_MIN > MIN_AVG_VOLUME

    def test_cdr_volume_min_is_500k(self):
        assert cdr_mapping.CDR_VOLUME_MIN == 500_000


class TestEnergyUnderlyings:
    def test_xom_is_energy(self):
        assert "XOM" in cdr_mapping.CDR_ENERGY_UNDERLYINGS

    def test_only_xom_remains(self):
        assert cdr_mapping.CDR_ENERGY_UNDERLYINGS == {"XOM"}


class TestGetUniverse:
    def test_tsx_returns_tsx60_only(self):
        result = cdr_mapping.get_universe("tsx")
        # All TSX tickers end with .TO or .V
        for t in result:
            assert ".NE" not in t, f"CDR ticker {t} found in tsx universe"

    def test_cdr_returns_ne_only(self):
        result = cdr_mapping.get_universe("cdr")
        for t in result:
            assert t.endswith(".NE"), f"{t} is not a .NE ticker in cdr universe"

    def test_tsx_cdr_contains_both(self):
        result = cdr_mapping.get_universe("tsx-cdr")
        has_tsx = any(t.endswith(".TO") for t in result)
        has_cdr = any(t.endswith(".NE") for t in result)
        assert has_tsx, "tsx-cdr universe has no .TO tickers"
        assert has_cdr, "tsx-cdr universe has no .NE tickers"

    def test_tsx_cdr_no_duplicates(self):
        result = cdr_mapping.get_universe("tsx-cdr")
        assert len(result) == len(set(result))

    def test_tsx_cdr_size_is_tsx_plus_cdr(self):
        tsx = cdr_mapping.get_universe("tsx")
        cdr = cdr_mapping.get_universe("cdr")
        combined = cdr_mapping.get_universe("tsx-cdr")
        assert len(combined) == len(set(tsx + cdr))

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown universe mode"):
            cdr_mapping.get_universe("invalid")
