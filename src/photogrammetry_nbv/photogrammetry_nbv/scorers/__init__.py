from .covisibility_scorer import CovisibilityScorer
from .weighted_sum_scorer import WeightedSumScorer
from .repair_weighted_covisibility_scorer import RepairWeightedCovisibilityScorer
from .baseline_aware_repair_weighted_covisibility_scorer import BaselineAwareRepairWeightedCovisibilityScorer
from .gt_phase_adaptive_hybrid_scorer import GTPhaseAdaptiveHybridScorer

SCORER_REGISTRY = {
    'weighted_sum': WeightedSumScorer,
    'covisibility': CovisibilityScorer,
    'repair_weighted_covisibility': RepairWeightedCovisibilityScorer,
    'baseline_aware_repair_weighted_covisibility': BaselineAwareRepairWeightedCovisibilityScorer,
    'gt_phase_adaptive_hybrid': GTPhaseAdaptiveHybridScorer,
}
