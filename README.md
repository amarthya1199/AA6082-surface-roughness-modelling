# AA6082 end milling – RSM vs ANN for surface roughness under leave-one-out validation

This repository holds the code and data behind the paper

> P. SVKSV Krishna Kiran, A. Singampalli, N. Kameswara Sarma. *Surface roughness modelling in CNC end milling of
> AA6082 from a 15-run Box–Behnken design: response surface and neural-network models compared under
> leave-one-out validation* (manuscript under review).

The scripts re-analyse a 15-run Box–Behnken experiment that varied spindle speed (1500–4500 rpm), feed rate
(200–500 mm/min) and depth of cut (0.2–0.6 mm). They produce every number, table and figure in the paper.

## Contents

```
data/aa6082_bbd_surface_roughness.csv   the 15 runs (N, f, d, Ra); provenance in data/README.md
src/analysis.py                         RSM (full + reduced quadratic, Type III ANOVA, lack of fit, PRESS),
                                        LOOCV of RSM and 3-h-1 ANNs, nested LOOCV, optimum + 95% PI
src/figures.py                          Figures 1–5 of the paper (600 dpi PNG + vector PDF)
results/                                outputs of a reference run (results.json, predictions.csv, figures/)
```

## How to reproduce

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/analysis.py data/aa6082_bbd_surface_roughness.csv results   # about 3 min on a laptop
python src/figures.py results
```

The scripts were tested with Python 3.13 and the package versions pinned in `requirements.txt`. The ANN fits use
fixed random seeds, so a rerun with these versions reproduces `results/` exactly. Other versions may differ in
the last digits.

## Methods in brief

- **RSM:** ordinary least squares in coded variables, a full quadratic model plus a reduced hierarchical model,
  validated with PRESS / predicted R² and lack of fit against the pure error from the centre replicates.
- **ANN:** 3-h-1 feed-forward networks (tanh hidden layer, linear output) trained with Levenberg–Marquardt
  (`scipy.optimize.least_squares`, `method="lm"`, analytic Jacobian), with weight decay λ added as augmented
  residuals. Each setting is run from ten random initialisations.
- **Validation:** leave-one-out cross-validation for every model. For the ANN, h and λ are chosen in an inner
  LOOCV loop that sees only the 14 training runs of each outer fold (nested LOOCV).

## Licence

Code: MIT (see `LICENSE`). Data: CC BY 4.0, with acknowledgement of the original dissertation (see `data/README.md`).
