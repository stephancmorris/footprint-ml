"""Tests for footprint_ml.encoders — ZoneEncoder."""

from __future__ import annotations

import math

import numpy as np
import pytest

from footprint_ml.encoders import ZoneEncoder


class TestZoneEncoderFit:
    def test_fit_returns_self(self) -> None:
        enc = ZoneEncoder()
        result = enc.fit(["IND", "B2"])
        assert result is enc

    def test_is_fitted_after_fit(self) -> None:
        enc = ZoneEncoder()
        assert not enc.is_fitted
        enc.fit(["IND"])
        assert enc.is_fitted

    def test_categories_sorted(self) -> None:
        enc = ZoneEncoder()
        enc.fit(["MU", "IND", "B2"])
        assert enc.categories == ["B2", "IND", "MU"]

    def test_none_excluded_from_categories(self) -> None:
        enc = ZoneEncoder()
        enc.fit(["IND", None, "B2"])
        assert None not in enc.categories
        assert len(enc.categories) == 2

    def test_duplicates_deduplicated(self) -> None:
        enc = ZoneEncoder()
        enc.fit(["IND", "IND", "B2"])
        assert enc.categories == ["B2", "IND"]

    def test_case_normalised_in_categories(self) -> None:
        enc = ZoneEncoder()
        enc.fit(["ind", "IND", "b2"])
        # After upper() dedup: {"IND", "B2"}
        assert "IND" in enc.categories
        assert "B2" in enc.categories
        assert len(enc.categories) == 2


class TestZoneEncoderTransform:
    def _fitted(self, codes: list[str | None] | None = None) -> ZoneEncoder:
        enc = ZoneEncoder()
        enc.fit(codes or ["IND", "B2", "MU"])
        return enc

    def test_transform_before_fit_raises(self) -> None:
        enc = ZoneEncoder()
        with pytest.raises(RuntimeError):
            enc.transform(["IND"])

    def test_returns_ndarray(self) -> None:
        enc = self._fitted()
        result = enc.transform(["IND"])
        assert isinstance(result, np.ndarray)

    def test_output_length_matches_input(self) -> None:
        enc = self._fitted()
        result = enc.transform(["IND", "B2", "MU"])
        assert len(result) == 3

    def test_known_codes_are_positive(self) -> None:
        enc = self._fitted()
        result = enc.transform(["IND", "B2"])
        assert all(v >= 0.0 for v in result)

    def test_unknown_code_maps_to_minus_one(self) -> None:
        enc = self._fitted()
        result = enc.transform(["UNKNOWN_ZONE"])
        assert result[0] == -1

    def test_none_maps_to_nan(self) -> None:
        enc = self._fitted()
        result = enc.transform([None])
        assert math.isnan(result[0])

    def test_case_insensitive_transform(self) -> None:
        enc = self._fitted(["IND", "B2"])
        upper = enc.transform(["IND"])
        lower = enc.transform(["ind"])
        assert upper[0] == lower[0]

    def test_whitespace_stripped_in_transform(self) -> None:
        enc = self._fitted(["IND", "B2"])
        plain = enc.transform(["IND"])
        spaced = enc.transform(["  IND  "])
        assert plain[0] == spaced[0]

    def test_same_code_same_value(self) -> None:
        enc = self._fitted()
        a = enc.transform(["IND"])
        b = enc.transform(["IND"])
        assert a[0] == b[0]

    def test_different_codes_different_values(self) -> None:
        enc = self._fitted()
        result = enc.transform(["IND", "B2", "MU"])
        # All three distinct codes must map to distinct values
        assert len(set(result.tolist())) == 3


class TestZoneEncoderFitTransform:
    def test_fit_transform_equivalent(self) -> None:
        enc_a = ZoneEncoder()
        enc_a.fit(["IND", "B2"])
        expected = enc_a.transform(["IND", "B2"])

        enc_b = ZoneEncoder()
        result = enc_b.fit_transform(["IND", "B2"])

        np.testing.assert_array_equal(expected, result)

    def test_fit_transform_includes_none(self) -> None:
        enc = ZoneEncoder()
        result = enc.fit_transform(["IND", None, "B2"])
        assert math.isnan(result[1])
