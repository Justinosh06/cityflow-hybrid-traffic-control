Hybrid RF and Q-learning
========================

The hybrid traffic-signal policy uses an LQF expert to train a Random Forest
(RF), then uses the RF probabilities to initialise first-visit Q-table states.
Q-learning continues to update those values from the configured traffic reward.

From the repository root, run the pipeline in order:

.. code-block:: console

    python -m src.collector
    python -m src.pretrainer
    python -m src.train --init none --q-table models/q_table_cold.joblib --output-csv data/training_metrics_cold.csv --seed 0 --episodes 500
    python -m src.train --init rf --q-table models/q_table.joblib --output-csv data/training_metrics_hybrid.csv --seed 0 --episodes 500
    python -m src.evaluate

The collector writes ``data/lqf_dataset.csv`` and the sorted lane-name list
``data/lqf_dataset_lanes.json``. The pretrainer writes
``models/rf_pretrain.joblib`` with the classifier, feature order, and discounted
initial value. Both training runs use the same seeded uncertainty scenarios.
Evaluation applies fixed-time, RF-only, cold-start Q-learning, and hybrid
RF-initialised Q-learning policies to the same five evaluation scenarios. It
writes per-scenario and mean metrics to ``data/evaluation_metrics.csv`` and the
RF-versus-hybrid action confusion matrix to
``data/evaluation_confusion_matrix.csv``.
