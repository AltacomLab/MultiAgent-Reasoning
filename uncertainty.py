import numpy as np


def prediction_uncertainty(predictions):
    """
    Variance across agent outputs
    """
    preds = np.vstack(predictions)
    return np.var(preds, axis=0)


def confidence_weights(uncertainty, eps=1e-6):
    """
    Inverse uncertainty weighting
    """
    return 1.0 / (uncertainty + eps)