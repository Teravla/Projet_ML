"""Calibration des probabilités (Platt, Isotonic, Temperature Scaling)."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression


def calibrate_classifier(
	base_model: LogisticRegression,
	method: str = "sigmoid",
	cv: int = 3,
) -> CalibratedClassifierCV:
	"""Crée un calibrateur sklearn (sigmoid=Platt, isotonic=Isotonic)."""

	if method not in {"sigmoid", "isotonic"}:
		raise ValueError("method doit être 'sigmoid' ou 'isotonic'")
	return CalibratedClassifierCV(estimator=base_model, method=method, cv=cv)


def summarize_confidence_distribution(max_probabilities: np.ndarray) -> dict[str, float]:
	"""Produit une synthèse des niveaux de confiance."""

	return {
		"p25": float(np.percentile(max_probabilities, 25)),
		"p50": float(np.percentile(max_probabilities, 50)),
		"p75": float(np.percentile(max_probabilities, 75)),
		"mean": float(np.mean(max_probabilities)),
		"std": float(np.std(max_probabilities)),
	}


def analyze_uncertain_predictions(max_probabilities: np.ndarray, threshold: float = 0.7) -> dict[str, float]:
	"""Calcule la part de prédictions incertaines pour un seuil donné."""

	uncertain_mask = max_probabilities < threshold
	uncertain_count = int(np.sum(uncertain_mask))
	total = int(max_probabilities.shape[0])
	uncertain_ratio = float(uncertain_count / total) if total else 0.0
	return {
		"threshold": float(threshold),
		"uncertain_count": uncertain_count,
		"total": total,
		"uncertain_ratio": uncertain_ratio,
	}
