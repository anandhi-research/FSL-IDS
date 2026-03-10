import numpy as np

def normalize_features(X):

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-8

    return (X - mean) / std


def quantize_weights(weights, bits=6):

    q_levels = 2 ** bits

    w_min = weights.min()
    w_max = weights.max()

    scale = (w_max - w_min) / (q_levels - 1)

    quantized = np.round((weights - w_min) / scale)

    return quantized * scale + w_min
