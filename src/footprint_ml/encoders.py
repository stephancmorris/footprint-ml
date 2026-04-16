"""Categorical encoding for zone codes, OSM tags, and ANZSIC divisions.

ZoneEncoder wraps scikit-learn's OrdinalEncoder and handles unseen categories
gracefully by mapping them to a dedicated "unknown" code rather than raising.
It is fitted during training and serialised as part of the model artifact.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import OrdinalEncoder


class ZoneEncoder:
    """Fit/transform zone code strings → stable ordinal floats.

    Unknown categories seen at inference time are mapped to ``-1.0`` so the
    downstream HistGradientBoostingClassifier can treat them as a distinct
    "unknown zone" bin rather than raising an error.

    Usage::

        enc = ZoneEncoder()
        enc.fit(["IND", "B2", "MU", None])
        enc.transform(["IND", "NEW_CODE", None])
        # → [1.0, -1.0, nan]
    """

    _UNKNOWN_CODE: int = -1

    def __init__(self) -> None:
        self._encoder: OrdinalEncoder | None = None
        self._categories: list[str] = []

    def fit(self, zone_codes: list[str | None]) -> "ZoneEncoder":
        """Fit the encoder on a list of zone code strings.

        ``None`` values are ignored during fitting (they map to NaN at
        transform time).
        """
        known = sorted({z.upper().strip() for z in zone_codes if z is not None})
        self._categories = known
        self._encoder = OrdinalEncoder(
            categories=[known],
            handle_unknown="use_encoded_value",
            unknown_value=self._UNKNOWN_CODE,
            encoded_missing_value=float("nan"),
            dtype=np.float64,
        )
        # OrdinalEncoder expects a 2-D array
        self._encoder.fit(np.array(known).reshape(-1, 1))
        return self

    def transform(self, zone_codes: list[str | None]) -> NDArray[np.float64]:
        """Encode a list of zone codes to a 1-D float array.

        ``None`` → ``nan``, unseen categories → ``-1``.
        """
        if self._encoder is None:
            raise RuntimeError("ZoneEncoder must be fitted before transform().")

        none_mask = [z is None for z in zone_codes]
        col: list[str] = [
            z.upper().strip() if z is not None else "__MISSING__" for z in zone_codes
        ]
        arr = np.array(col, dtype=object).reshape(-1, 1)
        result: NDArray[np.float64] = self._encoder.transform(arr).ravel().astype(np.float64)
        # Replace entries that were None with NaN (sklearn encodes them as unknown=-1)
        for i, is_none in enumerate(none_mask):
            if is_none:
                result[i] = float("nan")
        return result

    def fit_transform(self, zone_codes: list[str | None]) -> NDArray[np.float64]:
        """Fit and transform in one step."""
        return self.fit(zone_codes).transform(zone_codes)

    @property
    def categories(self) -> list[str]:
        """Sorted list of known zone code strings seen during fitting."""
        return list(self._categories)

    @property
    def is_fitted(self) -> bool:
        return self._encoder is not None
