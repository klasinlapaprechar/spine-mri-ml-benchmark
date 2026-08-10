# Experiment design

Subject-safe partitions reused from locked label CSVs (no re-split). K-fold is nested inside train; val drives early stopping; test scored once. External spine-generic evaluation is type-only (VOI/ACQ not informative on that corpus). Separate vs multi-head heads share the same preprocessing and split IDs so architecture comparisons are not confounded by data order.
