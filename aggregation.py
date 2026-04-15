import numpy as np


def majority_voting(predictions):
    preds = np.vstack(predictions)
    return np.mean(preds, axis=0)


def weighted_aggregation(predictions, weights):
    preds = np.vstack(predictions)
    w = weights.reshape(1, -1)
    return np.average(preds, axis=0, weights=weights)


def uncertainty_aware_aggregation(predictions):
    preds = np.vstack(predictions)

    var = np.var(preds, axis=0)
    weights = 1.0 / (var + 1e-6)

    final = []
    for i in range(preds.shape[1]):
        final.append(np.average(preds[:, i], weights=np.ones(preds.shape[0])))

    return np.array(final)