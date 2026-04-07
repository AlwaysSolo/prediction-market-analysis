from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class PlattCalibration:
    a: float
    b: float
    clip_min: float = 1e-6
    clip_max: float = 1.0 - 1e-6

    def apply(self, probabilities: np.ndarray | float) -> np.ndarray | float:
        clipped = np.clip(np.asarray(probabilities, dtype=np.float64), self.clip_min, self.clip_max)
        logits = np.log(clipped / (1.0 - clipped))
        calibrated = 1.0 / (1.0 + np.exp(-(self.a * logits + self.b)))
        calibrated = np.clip(calibrated, self.clip_min, self.clip_max)
        if np.isscalar(probabilities):
            return float(calibrated.item())
        return calibrated


def fit_platt_scaler(
    raw_probabilities: np.ndarray,
    actuals: np.ndarray,
    *,
    clip_min: float = 1e-6,
    clip_max: float = 1.0 - 1e-6,
) -> PlattCalibration:
    clipped = np.clip(np.asarray(raw_probabilities, dtype=np.float64), clip_min, clip_max)
    actuals_array = np.asarray(actuals, dtype=np.int8)
    if np.unique(actuals_array).size < 2:
        return PlattCalibration(a=1.0, b=0.0, clip_min=clip_min, clip_max=clip_max)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(solver="lbfgs", random_state=42, max_iter=1000)
    model.fit(logits, actuals_array)
    return PlattCalibration(
        a=float(model.coef_[0][0]),
        b=float(model.intercept_[0]),
        clip_min=clip_min,
        clip_max=clip_max,
    )


def load_platt_calibration(path: str | Path) -> PlattCalibration:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return PlattCalibration(
        a=float(payload["a"]),
        b=float(payload["b"]),
        clip_min=float(payload.get("clip_min", 1e-6)),
        clip_max=float(payload.get("clip_max", 1.0 - 1e-6)),
    )


def save_platt_calibration(path: str | Path, calibration: PlattCalibration) -> None:
    payload = {
        "method": "platt",
        "a": calibration.a,
        "b": calibration.b,
        "clip_min": calibration.clip_min,
        "clip_max": calibration.clip_max,
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
