"""footprint-ml: ML classifier for building footprint polygons → commercial property asset classes."""

from footprint_ml.types import AssetClass, Prediction

__version__ = "0.1.0"

# Populated as modules are implemented:
# from footprint_ml.classifier import FootprintClassifier
# from footprint_ml.trainer import FootprintTrainer

__all__ = ["__version__", "AssetClass", "Prediction"]
