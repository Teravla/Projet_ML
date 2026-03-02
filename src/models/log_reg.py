"""Régression logistique multinomiale."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


@dataclass(frozen=True)
class PredictionSummary:
	"""Résumé des prédictions probabilistes."""

	y_pred: np.ndarray
	max_prob: np.ndarray
	uncertain_mask: np.ndarray


def flatten_images(images: np.ndarray) -> np.ndarray:
	"""Aplati un batch d'images (N,H,W,C) vers (N,F)."""

	if images.ndim < 2:
		raise ValueError("Le tenseur images est invalide")
	return images.reshape(images.shape[0], -1)


def build_logistic_regression(
	max_iter: int = 600,
	random_state: int = 42,
) -> LogisticRegression:
	"""Construit une régression logistique multinomiale."""

	return LogisticRegression(
		solver="lbfgs",
		max_iter=max_iter,
		random_state=random_state,
	)


def train_logistic_regression(
	x_train: np.ndarray,
	y_train: np.ndarray,
	max_iter: int = 600,
	random_state: int = 42,
) -> LogisticRegression:
	"""Entraîne le modèle de régression logistique multinomiale."""

	model = build_logistic_regression(max_iter=max_iter, random_state=random_state)
	model.fit(x_train, y_train)
	return model


def predict_with_confidence(
	model: LogisticRegression,
	x_data: np.ndarray,
	uncertainty_threshold: float = 0.7,
) -> PredictionSummary:
	"""Retourne classes prédites, confiance max et masque d'incertitude."""

	probas = model.predict_proba(x_data)
	y_pred = np.argmax(probas, axis=1)
	max_prob = np.max(probas, axis=1)
	uncertain_mask = max_prob < uncertainty_threshold
	return PredictionSummary(y_pred=y_pred, max_prob=max_prob, uncertain_mask=uncertain_mask)


def evaluate_model(model: LogisticRegression, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, object]:
	"""Calcule les métriques standards pour un modèle sklearn."""

	y_pred = model.predict(x_test)
	accuracy = accuracy_score(y_test, y_pred)
	report = classification_report(y_test, y_pred, output_dict=True)
	return {"accuracy": float(accuracy), "report": report}
