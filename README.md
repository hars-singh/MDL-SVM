# MDL-SVM

Python implementation of linear SVM model selection for [Data Efficient Support Vector Machine Training Using the Minimum Description Length Principle](https://ieeexplore.ieee.org/abstract/document/9747649).

## Quick start

Requires Python 3.9 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python mdl_svm.py --c-values 0.1 1 10 --output results/demo.json
```

The demo uses scikit-learn's bundled Wisconsin Diagnostic Breast Cancer dataset and needs no CSV download. It holds out a stratified 20% test set, fits preprocessing on training data, and selects `C` using training MDL only. The test set is used for final reporting. Malignant is the positive class, matching the notebook.

Run the notebook's full `C` grid and an optional four-fold accuracy-based cross-validation baseline:

```bash
python mdl_svm.py --compare-cv --output results/comparison.json
# Equivalent installed command:
mdl-svm --c-values 0.1 1 10 --compare-cv
```

Large values of `C` can take longer to fit. Both methods use the same held-out test set; cross-validation preprocessing is fitted separately within each training fold.

## Python API

```python
from mdl_svm import fit_mdl_svm

# X_train/y_train contain training observations only.
selection = fit_mdl_svm(X_train, y_train, c_values=[0.1, 1, 10])
print(selection.best)
predictions = selection.estimator.predict(X_test)
```

Inputs must be finite numeric features and binary labels. The returned scikit-learn pipeline stores the training-fitted scaler and selected linear SVC. Candidate statistics are available in `selection.candidates`.

## Criterion

For each candidate, compute signed geometric distances `d = decision_function(X) / ||w||`, the population variance `variance(d)`, and:

```text
MDL = 0.5 * N_support * log2(N)
    + 0.5 * N * log2(2 * pi * e * variance(d))
```

The candidate with the smallest MDL is retained; ties use input order.

## Citation

```bibtex
@inproceedings{singh2022data,
  title={Data Efficient Support Vector Machine Training Using the Minimum Description Length Principle},
  author={Singh, Harsh and Arandjelovi\'{c}, Ognjen},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1361--1365},
  year={2022},
  doi={10.1109/ICASSP43922.2022.9747649}
}
```
