"""Linear SVM model selection using minimum description length.

Converted from Linear_SVM_version.ipynb. Implements its Gaussian criterion
on signed geometric distances to the separating hyperplane.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_X_y


DEFAULT_C = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


@dataclass(frozen=True)
class Candidate:
    C: float
    support_vectors: int
    distance_variance: float
    mdl_bits: float


@dataclass
class Selection:
    estimator: object
    best: Candidate
    candidates: list


def gaussian_mdl(distances, n_support, variance_floor=1e-12):
    """Return N_sv/2 log2(N) + N/2 log2(2*pi*e*variance).

    Variance is the maximum-likelihood estimate (ddof=0), as in the notebook.
    A variance floor handles degenerate distances; differential code lengths
    may legitimately be negative. They should only be compared on the same
    training data, preprocessing, and distance convention.
    """
    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 1 or distances.size < 2:
        raise ValueError("Provide a one-dimensional array of at least two distances.")
    if not np.isfinite(distances).all():
        raise ValueError("Distances must be finite.")
    if not isinstance(n_support, (int, np.integer)) or not 1 <= n_support <= distances.size:
        raise ValueError("n_support must be an integer between 1 and N.")
    if not np.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("variance_floor must be finite and positive.")
    variance = max(float(np.var(distances)), variance_floor)
    n = distances.size
    return float(0.5 * n_support * np.log2(n)
                 + 0.5 * n * np.log2(2 * np.pi * np.e * variance))


def fit_mdl_svm(X, y, c_values=DEFAULT_C, variance_floor=1e-12):
    """Fit binary linear SVM candidates and select the smallest training MDL.

    Pass training data only. The returned pipeline applies its training-fitted
    StandardScaler to new observations. No validation/test data is consulted.
    """
    X, y = check_X_y(X, y, dtype=float)
    if np.unique(y).size != 2:
        raise ValueError("This notebook-derived implementation requires two classes.")
    c_values = tuple(float(c) for c in c_values)
    if not c_values or any(not np.isfinite(c) or c <= 0 for c in c_values):
        raise ValueError("c_values must contain finite, positive values.")
    if not np.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("variance_floor must be finite and positive.")
    scaler = StandardScaler().fit(X)
    scaled = scaler.transform(X)
    rows = []
    best_model = None
    best_row = None
    for c in c_values:
        model = SVC(kernel="linear", C=c).fit(scaled, y)
        norm = float(np.linalg.norm(model.coef_))
        if not np.isfinite(norm) or norm <= np.finfo(float).eps:
            raise ValueError("A candidate has a degenerate separating hyperplane.")
        distances = model.decision_function(scaled) / norm
        n_support = int(model.support_.size)
        row = Candidate(c, n_support, float(np.var(distances)),
                        gaussian_mdl(distances, n_support, variance_floor))
        rows.append(row)
        if best_row is None or row.mdl_bits < best_row.mdl_bits:
            best_model, best_row = model, row
    return Selection(make_pipeline(scaler, best_model), best_row, rows)


def _metrics(estimator, X, y):
    prediction = estimator.predict(X)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
    }


def run_experiment(c_values=DEFAULT_C, seed=42, test_size=0.2, compare_cv=False):
    """Run a reproducible demo on scikit-learn's bundled WDBC dataset.

    A held-out test set is used only after selection. The optional four-fold
    CV baseline fits preprocessing independently inside every training fold.
    """
    data = load_breast_cancer()
    # Match the notebook: malignant=1, benign=0 (sklearn uses the reverse).
    y = (data.target == 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, y, test_size=test_size, random_state=seed, stratify=y)
    selection = fit_mdl_svm(X_train, y_train, c_values)
    report = {
        "variant": "notebook-derived linear SVM / Gaussian MDL",
        "seed": seed,
        "training_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "positive_class": "malignant",
        "best_candidate": asdict(selection.best),
        "candidates": [asdict(row) for row in selection.candidates],
        "test_metrics": _metrics(selection.estimator, X_test, y_test),
    }
    if compare_cv:
        cv = GridSearchCV(
            make_pipeline(StandardScaler(), SVC(kernel="linear")),
            {"svc__C": list(c_values)},
            cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=seed),
            scoring="accuracy", refit=True)
        cv.fit(X_train, y_train)
        report["cross_validation_baseline"] = {
            "best_C": float(cv.best_params_["svc__C"]),
            "test_metrics": _metrics(cv.best_estimator_, X_test, y_test),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c-values", nargs="+", type=float, default=DEFAULT_C)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--compare-cv", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional JSON results file.")
    args = parser.parse_args()
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be strictly between 0 and 1")
    report = run_experiment(args.c_values, args.seed, args.test_size, args.compare_cv)
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
