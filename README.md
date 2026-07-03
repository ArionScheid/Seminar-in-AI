# GNNExplainer Seminar Project

Seminar report and slides on *GNNExplainer: Generating Explanations for Graph
Neural Networks* (Ying et al., NeurIPS 2019), with a companion Python script
implementing the experiments discussed in the report.

## Files

- `reportv2.tex` — the seminar report.
- `slidesAI.tex` — accompanying presentation slides (Beamer).
- `references.bib` — shared bibliography for both.
- `gnnexplainer.py` — experiment script for the report's multi-size
  clique detection and color-anchor benchmark.

## Running the experiments

```
pip install torch torch_geometric networkx numpy
python gnnexplainer.py
```

Tested with Python 3.13, PyTorch 2.12, PyTorch Geometric 2.8. Takes about a
minute on CPU. The random seed is fixed (42) for reproducibility.
