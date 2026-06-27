# Wasserstein Spatial Depth for Abnormal Batch Detection

Code for the paper **"A Wasserstein Framework for Abnormal Batch Detection in Streaming Data under Distributional Heterogeneity"** (STAI-X 2026).

Each data batch is treated as a distribution in Wasserstein-2 space. We score a batch's centrality with **Wasserstein Spatial Depth (WSD)** and convert the depth into an empirical p-value — the rank of the batch's depth among a sliding window of reference batches. Batches whose p-value falls below a level $\alpha$ are flagged as abnormal and can be filtered during online SGD. The p-value is approximately $\mathrm{Uniform}[0,1]$ for regular batches, giving a distribution-adaptive cutoff that respects natural batch-to-batch heterogeneity.

## Contents

- `STAIX_2026_tables.ipynb` — a single self-contained notebook that reproduces every numerical result in the paper.
- `requirements.txt` — pinned package versions.

## Setup

CPU only; no GPU or specialized hardware required. Tested with Python 3.11.

```bash
pip install -r requirements.txt
```

MNIST (Section 5) is downloaded automatically by `torchvision` on first run and cached locally; no manual download step.

## Running

Open `STAIX_2026_tables.ipynb` and **Restart & Run All**. To run a single experiment, execute *Section 0* (environment) and *Section 1* (shared WSD core) first, then that experiment's setup cell followed by its table/figure cells.

| Section | Experiment | Outputs |
|---|---|---|
| §4.1 | Two-sample testing — four synthetic cases (location drift / scale / rotation / translation) | Tables 1, 4, 5–8 |
| §4.2 | Online-regression OSGD with intra-batch $\theta$-jitter anomalies | Tables 2, 9, 10; Figures 3–4 |
| §5 | Streaming-MNIST CNN with label-flip anomalies | Tables 3, 11, 12; Figures 5–6 |

Each experiment compares the WSD-based p-value against standard baselines: moving average (MA), QuantTree, energy distance, and MMD for §4.1; gradient norm, Mean-grad $L_2$, Hotelling $T^2$, and MMD for §4.2 and §5.

## Reproducibility notes

- All optimal-transport plans use **exact EMD** (`ot.emd` from the POT library) — no entropic / Sinkhorn approximation.
- All reported numbers are mean ± s.d. over **20 independent seeds** (0–19); results are reproducible up to negligible Monte-Carlo variation in the last digit.
- Per-run cost is modest on a standard CPU (see Table 13 in the paper).

## Citation

```bibtex
@inproceedings{xu2026wasserstein,
  title     = {A Wasserstein Framework for Abnormal Batch Detection in
               Streaming Data under Distributional Heterogeneity},
  author    = {Xu, Chenqin and Yao, Yisha},
  booktitle = {STAI-X},
  year      = {2026}
}
```
