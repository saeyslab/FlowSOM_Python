from sklearn.metrics import v_measure_score

from flowsom.main import FlowSOM
from flowsom.models import FlowSOMEstimator


def test_clustering(X):
    fsom = FlowSOMEstimator(n_clusters=10)
    y_pred = fsom.fit_predict(X)
    assert y_pred.shape == (100,)


def test_clustering_v_measure(X_and_y):
    som = FlowSOMEstimator(n_clusters=10)
    X, y_true = X_and_y
    y_pred = som.fit_predict(X)
    score = v_measure_score(y_true, y_pred)
    assert score > 0.7


def test_reproducibility_no_seed(X):
    fsom_1 = FlowSOMEstimator(n_clusters=10)
    fsom_2 = FlowSOMEstimator(n_clusters=10)
    y_pred_1 = fsom_1.fit_predict(X)
    y_pred_2 = fsom_2.fit_predict(X)

    assert not all(y_pred_1 == y_pred_2)


def test_reproducibility_seed(X):
    fsom_1 = FlowSOMEstimator(n_clusters=10, seed=0)
    fsom_2 = FlowSOMEstimator(n_clusters=10, seed=0)
    y_pred_1 = fsom_1.fit_predict(X)
    y_pred_2 = fsom_2.fit_predict(X)

    assert all(y_pred_1 == y_pred_2)


def test_metacluster(X):
    fsom = FlowSOM(X, n_clusters=10)
    assert 10 == fsom.model.metacluster_model.n_clusters
    fsom.metacluster(n_clusters=5)
    assert 5 == fsom.model.metacluster_model.n_clusters
