import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from mdl_svm import fit_mdl_svm, gaussian_mdl, run_experiment


def test_gaussian_code_length_matches_log_likelihood():
    distances = np.array([-2.0, -0.5, 1.0, 3.0])
    variance = np.var(distances)
    log_density = (-0.5 * np.log(2 * np.pi * variance)
                   - (distances - distances.mean()) ** 2 / (2 * variance))
    expected = 0.5 * 2 * np.log2(len(distances)) - log_density.sum() / np.log(2)
    assert gaussian_mdl(distances, 2) == pytest.approx(expected)


def test_constant_distances_have_finite_code_length():
    assert np.isfinite(gaussian_mdl(np.ones(5), 2))


@pytest.mark.parametrize("distances,support", [([1], 1), ([1, np.nan], 1), ([1, 2], 0)])
def test_invalid_distance_inputs_rejected(distances, support):
    with pytest.raises(ValueError):
        gaussian_mdl(distances, support)


def test_selection_uses_training_distances_and_fits_scaler_on_training_only():
    X, y = make_classification(n_samples=70, n_features=5, random_state=4)
    selection = fit_mdl_svm(X, y, [0.1, 1.0])
    np.testing.assert_allclose(selection.estimator[0].mean_, X.mean(axis=0))
    scores = []
    scaled = StandardScaler().fit_transform(X)
    for c in [0.1, 1.0]:
        svc = SVC(kernel="linear", C=c).fit(scaled, y)
        distances = svc.decision_function(scaled) / np.linalg.norm(svc.coef_)
        scores.append(gaussian_mdl(distances, int(svc.support_.size)))
    assert [r.mdl_bits for r in selection.candidates] == pytest.approx(scores)
    assert selection.best.C == [0.1, 1.0][int(np.argmin(scores))]
    before = selection.estimator[0].mean_.copy()
    selection.estimator.predict(X + 100)
    np.testing.assert_array_equal(selection.estimator[0].mean_, before)


@pytest.mark.parametrize("values", [[], [0], [-1], [float("nan")]])
def test_invalid_candidate_parameters_rejected(values):
    with pytest.raises(ValueError):
        fit_mdl_svm([[0], [1], [2], [3]], [0, 0, 1, 1], values)


def test_multiclass_input_rejected():
    with pytest.raises(ValueError, match="two classes"):
        fit_mdl_svm([[0], [1], [2]], [0, 1, 2])


def test_experiment_has_disjoint_training_and_test_counts():
    result = run_experiment(c_values=[0.1, 1.0], compare_cv=True)
    assert result["training_samples"] == 455
    assert result["test_samples"] == 114
    assert result["positive_class"] == "malignant"
    assert 0 <= result["test_metrics"]["accuracy"] <= 1
    assert "cross_validation_baseline" in result
