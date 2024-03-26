from sklearn.metrics import v_measure_score

from flowsom.models import SOMEstimator


def test_clustering(X):
    som = SOMEstimator()
    y_pred: SOMEstimator = som.fit_predict(X)
    assert y_pred.shape == (100,)


def test_clustering_v_measure(X_and_y):
    som = SOMEstimator()
    X, y_true = X_and_y
    y_pred = som.fit_predict(X)
    score = v_measure_score(y_true, y_pred)
    assert score > 0.7
