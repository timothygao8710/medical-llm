import numpy as np

# @ToDO: experiement with different formulas - regular entropy?
def predictive_entropy_rao(log_probs):
    entropy = -np.sum(np.exp(log_probs) * log_probs)
    return entropy

def get_entropy_from_probabilities(probabilities):
    assert np.isclose(probabilities.sum(), 1, rtol=1.e-3)
    entropy = - (probabilities * np.log(probabilities)).sum()
    return entropy

def cluster_assignment_entropy(semantic_ids):
    """Estimate semantic uncertainty from how often different clusters get assigned.

    We estimate the categorical distribution over cluster assignments from the
    semantic ids. The uncertainty is then given by the entropy of that
    distribution. This estimate does not use token likelihoods, it relies soley
    on the cluster assignments. If probability mass is spread of between many
    clusters, entropy is larger. If probability mass is concentrated on a few
    clusters, entropy is small.

    Input:
        semantic_ids: List of semantic ids, e.g. [0, 1, 2, 1].
    Output:
        cluster_entropy: Entropy, e.g. (-p log p).sum() for p = [1/4, 2/4, 1/4].
    """

    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts/n_generations
    print(probabilities)
    return get_entropy_from_probabilities(probabilities)

# Given a list of string, compute the discrete semantic uncertainty. or cluster semantic uncertainty (see notebook)
def get_semantic_uncertainty(strings):
    from entailment import get_semantic_ids, EntailmentDeberta

    n = len(strings)
    strings = ["Paris is the capital of France", "France's capital is Paris", "When someone visits France, they go to Paris", "Random"]

    model = EntailmentDeberta()
    classes = get_semantic_ids(strings, model)

    return cluster_assignment_entropy(classes)


def semantic_uncertainty_from_outputs(outputs):
    probs = np.array(get_probs_from_outputs(outputs))[:, 1]
    probs = probs.astype(float)
    return get_entropy_from_probabilities(probs)

def get_probs_from_outputs(arr):
    mp = {}
    for i in arr:
        if i not in mp:
            mp[i] = 0
        mp[i] += 1
    return sorted([[i, mp[i] / len(arr)] for i in mp.keys()], key=lambda x: -x[1])

if __name__ == '__main__':
    # strings = ["Paris is the capital of France", "France's capital is Paris", "When someone visits France, they go to Paris", "China's capital is Beijing", "Beijing is China's capital", "Random"]
    # Output: [0, 0, 1, 2, 2, 3]
    # res = get_semantic_uncertainty(strings)
    # print(res)
    
    print(semantic_uncertainty_from_outputs(['A', 'A', 'B', 'C', 'C', 'D']))

    