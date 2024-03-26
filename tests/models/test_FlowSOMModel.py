from flowsom.models import FlowSOMEstimator
from sklearn.metrics import v_measure_score

def test_clustering(X):
    fsom = FlowSOMEstimator(
        cluster_kwargs=dict(),
        metacluster_kwargs=dict(
            n_clusters=10)
    )
    y_pred = fsom.fit_predict(X)
    assert y_pred.shape == (100,)

def test_clustering_v_measure(X_and_y):
    som = FlowSOMEstimator(
        cluster_kwargs=dict(),
        metacluster_kwargs=dict(n_clusters=10)
    )
    X, y_true = X_and_y
    y_pred = som.fit_predict(X)
    score = v_measure_score(y_true, y_pred)
    assert score > 0.7