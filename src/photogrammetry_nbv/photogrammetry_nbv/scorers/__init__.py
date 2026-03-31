from .covisibility_scorer import CovisibilityScorer
from .weighted_sum_scorer import WeightedSumScorer

SCORER_REGISTRY = {
    'weighted_sum': WeightedSumScorer,
    'covisibility': CovisibilityScorer,
}
