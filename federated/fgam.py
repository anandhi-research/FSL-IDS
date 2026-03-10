import numpy as np

def fgam_aggregate(client_weights):

    aggregated_weights = []

    for weights in zip(*client_weights):

        weights = np.array(weights)

        variance = np.var(weights, axis=0)

        trust_scores = np.exp(-variance)

        weighted = weights * trust_scores

        aggregated = np.mean(weighted, axis=0)

        aggregated_weights.append(aggregated)

    return aggregated_weights
