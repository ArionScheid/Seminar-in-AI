# GNNExplainer, Revisited: A Flawed Bound, Local Optima, and Anchor Features

**Arion Scheid** · Seminar in Artificial Intelligence · TU Wien · 2026

📄 **[Read the report (PDF)](https://github.com/ArionScheid/Seminar-in-AI/blob/main/report.pdf)** · 🖥️ [Slides (PDF)](https://github.com/ArionScheid/Seminar-in-AI/blob/main/slides.pdf) · 💻 [Experiment code](gnnexplainer.py)

This is a critical study of *GNNExplainer: Generating Explanations for Graph
Neural Networks* (Ying et al., NeurIPS 2019), the most widely cited post-hoc
explainer for graph neural networks. The report derives the method from first
principles and finds an error in the original derivation. It then tests two
failure modes on a new synthetic benchmark with known ground-truth
explanations.

---

## Key contributions

### 1. A flawed step in the original derivation (Jensen's inequality)

Turning GNNExplainer's discrete subgraph search into a single differentiable
forward pass is its central step. Ying et al. treat the explanation subgraph
as a random variable $\mathcal{G}_S \sim \mathcal{Q}$ and swap the expected
conditional entropy for the entropy at the *expected* (soft-masked) graph:

$$
\mathbb{E}_{\mathcal{Q}}\big[H(Y \mid \mathcal{G}_S)\big]
\;\longrightarrow\;
H\big(Y \mid \mathbb{E}_{\mathcal{Q}}[\mathcal{G}_S]\big).
$$

The paper says that, "with convexity assumption", this substitute is an
**upper** bound on the original objective. **That is backwards.** For a convex
$\varphi$, Jensen's inequality gives
$\varphi(\mathbb{E}[Z]) \le \mathbb{E}[\varphi(Z)]$, so under the paper's own
assumption the substitute is a **lower** bound. Minimizing a lower bound tells
you nothing about the quantity you actually care about. The authors also admit
that the convexity assumption does not hold for neural networks. So the
continuous relaxation behind every GNNExplainer implementation is a heuristic,
not a provable bound. Its value comes from empirical results, not from the
derivation.
→ [Report, Section 4.5](https://github.com/ArionScheid/Seminar-in-AI/blob/main/report.pdf)

### 2. Characterizing local optima on a new multi-size clique benchmark

I built a benchmark that embeds 3-, 4- and 5-cliques in a sparse
Erdős–Rényi graph. On it, any sub-clique of the true clique is already a
locally consistent explanation. GNNExplainer (PyTorch Geometric) never
recovers a full clique: explanation F1 stays between **0.13 and 0.26** across
clique sizes and is worst at $k=5$. This is the sub-clique local-minimum
mechanism the original authors expected but never characterized.
→ Report, Sections 4.7 and 5.1

| Clique size | Precision | Recall | F1 | Largest recovered clique |
|---|---|---|---|---|
| $k=3$ | 0.202 | 0.326 | 0.199 ± 0.134 | 1.96 / 3 |
| $k=4$ | 0.286 | 0.329 | 0.261 ± 0.124 | 2.56 / 4 |
| $k=5$ | 0.156 | 0.117 | 0.132 ± 0.110 | 1.87 / 5 |

### 3. Testing a 2026 unfaithfulness result for post-hoc explainers

Azzolin et al. (2026) prove that *self-explainable* GNNs can be perfectly
accurate while their explanations have nothing to do with their reasoning.
They do this through "anchor" patterns that appear in every input. I adapted
this construction from graph-level anchor nodes to node-level anchor
*features*, and argued why the strong version of the failure cannot happen for
a post-hoc explainer. I then tested empirically for a weaker version by adding
two constant, label-independent color features. One feature shows almost no
class-dependent gap in mask weight (0.230 vs. 0.209). The other shows a large
gap (0.572 vs. 0.291), far larger than the classifier's accuracy difference
between the two classes would suggest.
→ Report, Sections 4.8 and 5.2

---

## Repository contents

| File | Description |
|---|---|
| [`report.pdf`](report.pdf) | The compiled seminar report (10 pages, NeurIPS style) |
| [`report.tex`](report.tex) | LaTeX source of the report |
| [`slides.pdf`](slides.pdf) | Compiled seminar presentation |
| [`slides.tex`](slides.tex) | Beamer source of the slides |
| [`gnnexplainer.py`](gnnexplainer.py) | Experiment script: dataset, GCN training, explainer evaluation |
| [`references.bib`](references.bib) | Bibliography shared by report and slides |
| [`neurips_2022.sty`](neurips_2022.sty) | NeurIPS 2022 style file, included so the report builds as-is |
| [`figs/`](figs) | Figures used in the slides |

## Reproducing the experiments

```bash
pip install torch torch_geometric networkx numpy
python gnnexplainer.py
```

Tested with Python 3.13, PyTorch 2.12 and PyTorch Geometric 2.8. It runs in
about a minute on a CPU, and the random seed is fixed (42) for
reproducibility.

## Building the PDFs

```bash
pdflatex report && bibtex report && pdflatex report && pdflatex report
pdflatex slides && pdflatex slides
```

## Citation

```bibtex
@misc{scheid2026gnnexplainer,
  author       = {Scheid, Arion},
  title        = {{GNNExplainer}: Generating Explanations for Graph Neural Networks},
  howpublished = {Seminar report, TU Wien},
  year         = {2026},
  url          = {https://github.com/ArionScheid/Seminar-in-AI}
}
```

## References

- R. Ying, D. Bourgeois, J. You, M. Zitnik, J. Leskovec. *GNNExplainer: Generating Explanations for Graph Neural Networks.* NeurIPS 2019.
- S. Azzolin, S. Teso, B. Lepri, A. Passerini, S. Malhotra. *GNN Explanations that do not Explain and How to Find Them.* ICLR 2026. [arXiv:2601.20815](https://arxiv.org/abs/2601.20815)
