from .weighted_sum_scorer import WeightedSumScorer

SCORER_REGISTRY = {
    'weighted_sum': WeightedSumScorer,
}
