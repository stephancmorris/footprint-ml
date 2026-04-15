"""Core types: AssetClass enum and Prediction dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AssetClass(str, Enum):
    """Commercial property asset classes.

    Values match Pulse's DB enum exactly — this is the authoritative list.
    """

    INDUSTRIAL = "industrial"
    WAREHOUSE = "warehouse"
    RETAIL = "retail"
    OFFICE = "office"
    MEDICAL = "medical"
    HOSPITALITY = "hospitality"
    EDUCATION = "education"
    CHILDCARE = "childcare"
    MIXED_USE = "mixed_use"
    OTHER_COMMERCIAL = "other_commercial"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Prediction:
    """Result of a single building footprint classification.

    Attributes:
        asset_class: The predicted asset class as a string (e.g. "warehouse").
        confidence: Probability of the top predicted class, in [0.0, 1.0].
        probabilities: Full probability distribution over all asset classes.
        model_version: Identifier of the model that produced this prediction.
    """

    asset_class: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str
