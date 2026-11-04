"""EK-FAC influence-function attribution for the per-document quality target.

Replaces the noisy difference-in-means per-doc target with a single-model
EK-FAC influence score (via the vendored `kronfluence` library), then refits
the interpretable EBM on the new target.

Pipeline (scripts live in ../scripts/):
  01_train_attribution_model.py   -> train the ~100M attribution model on the full train pool
  02_fit_factors.py       -> fit EK-FAC factors once
  03_score_influence.py   -> score docs, sharded over data partitions
  04_aggregate_influence.py -> reduce + sign + rank-transform -> EBM target
  05_influence_diagnostics.py -> reliability / vs-DiM / EBM comparison
"""
