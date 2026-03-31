# photogrammetry_nbv

Phase 2 package for sparse-point-cloud-driven active shape estimation.

## Main components

- `phase2_controller_node.py`
  - Online mission loop.
  - Boots a persistent Metashape project from the 4 seed images.
  - Generates Fibonacci-hemisphere candidates.
  - Filters and scores candidates.
  - Captures one image per iteration until the image budget is exhausted.
  - Returns home, lands, and launches the offline dense step.

- `offline_phase2_eval.py`
  - Offline validation tool.
  - Lets you test seed loading, sparse metrics ingestion, candidate generation, and scoring without moving the drone.

- `metashape_scripts/`
  - Headless scripts for project bootstrap, incremental update, sparse metrics export, and final dense reconstruction.

- `scorers/`
  - Plug-in scoring implementations.
  - The controller only depends on the `BaseScorer` contract.

## Scoring modularity

To swap the score function later:

1. Add a new scorer class under `photogrammetry_nbv/scorers/`.
2. Register it in `photogrammetry_nbv/scorers/__init__.py`.
3. Change `config/scoring.yaml`.

No controller changes should be necessary as long as the new scorer returns the same score payload format.
