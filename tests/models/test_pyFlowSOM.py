import pytest
from sklearn.metrics import v_measure_score

# optional import if pyFlowSOM is installed, otherwise use regular FlowSOM for type checking
try:
    from flowsom.models.pyflowsom_estimator import PyFlowSOMEstimator
except ImportError:
    from flowsom.models import FlowSOMEstimator as PyFlowSOMEstimator


@pytest.importorskip("flowsom.models.pyflowsom_estimator")
def test_clustering(X):
    fsom = PyFlowSOMEstimator(n_clusters=10)
    y_pred = fsom.fit_predict(X)
    assert y_pred.shape == (100,)


@pytest.importorskip("flowsom.models.pyflowsom_estimator")
def test_clustering_v_measure(X_and_y):
    som = PyFlowSOMEstimator(n_clusters=10)
    X, y_true = X_and_y
    y_pred = som.fit_predict(X)
    score = v_measure_score(y_true, y_pred)
    assert score > 0.7


@pytest.importorskip("flowsom.models.pyflowsom_estimator")
def test_reproducibility_no_seed(X):
    fsom_1 = PyFlowSOMEstimator(n_clusters=10)
    fsom_2 = PyFlowSOMEstimator(n_clusters=10)
    y_pred_1 = fsom_1.fit_predict(X)
    y_pred_2 = fsom_2.fit_predict(X)

    assert not all(y_pred_1 == y_pred_2)


@pytest.importorskip("flowsom.models.pyflowsom_estimator")
def test_reproducibility_seed(X):
    fsom_1 = PyFlowSOMEstimator(n_clusters=10, seed=0)
    fsom_2 = PyFlowSOMEstimator(n_clusters=10, seed=0)
    y_pred_1 = fsom_1.fit_predict(X)
    y_pred_2 = fsom_2.fit_predict(X)

    assert all(y_pred_1 == y_pred_2)
