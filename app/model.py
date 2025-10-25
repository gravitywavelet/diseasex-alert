import json, joblib
import numpy as np
from pathlib import Path
from .config import settings

class ModelService:
    """Wraps the trained sklearn pipeline for inference and metadata."""
    def __init__(self, model_path: Path, stats_path: Path):
        self.model_path = model_path
        self.stats_path = stats_path
        self.pipe = None
        self.stats = {}

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing model: {self.model_path}")
        self.pipe = joblib.load(self.model_path)
        if self.stats_path.exists():
            self.stats = json.loads(self.stats_path.read_text())
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.pipe.predict_proba(X)

    def cutoff_for_coverage(self, coverage: float):
        """Get precomputed cutoff from quantiles in training stats."""
        q = (self.stats or {}).get("p_treated_quantiles", {})
        key = f"{coverage:.2f}"
        if key in q:
            return float(q[key])
        if q:
            pairs = sorted((float(k), float(v)) for k, v in q.items())
            return min(pairs, key=lambda kv: abs(kv[0]-coverage))[1]
        return None

model_service = ModelService(settings.MODEL_PATH, settings.STATS_PATH).load()