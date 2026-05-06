# Explainable Machine Learning for Locational Frequency Stability Assessment

Repository for the BEng Individual Project *Explainable Machine Learning for Locational Frequency Stability Assessment* (EEEN30330), Department of Electrical and Electronic Engineering, The University of Manchester, 2025/26.

This repository is publicly accessible and is referenced from the appendix of the Final Report.

1. Introduction

This project trains and explains 60 machine-learning surrogate models for short-term frequency-stability assessment of a modified IEEE 9-bus system under N-1 generator-outage contingency. For each of the 9 buses the surrogates predict the local rate of change of frequency (RoCoF) and the frequency nadir; two additional system-wide worst-case targets bring the total to 20 targets. Three algorithms are trained per target (Multilayer Perceptron, Random Forest, XGBoost), giving 60 trained models in total. SHAP values are then computed for every model, summarised globally and locally, and compared across algorithms using Spearman rank correlation.

The work adapts and extends Kilembe, Hamilton, and Papadopoulos [1] by extending the SHAP analysis from a single neural-network surrogate to all three algorithms, and by introducing a cross-algorithm consistency check on the SHAP-based feature rankings. This consistency check is the original methodological contribution of the project.

2. Overview

The pipeline runs end-to-end from a single CSV file:


CSV
  -> preprocessing (clean, ODD filter, stratified split, scaling)
  -> per-target training (3 algorithms x 20 targets = 60 models)
  -> SHAP value computation (Tree for RF/XGB, Kernel for MLP)
  -> global SHAP analysis (bar, beeswarm, heatmaps, dependence)
  -> local SHAP analysis (waterfall plots for critical scenarios)
  -> cross-algorithm SHAP consistency (Spearman rho across algorithms)


Inputs (9 features): `SG 1 MVA`, `SG 2 MVA`, `SG 3 MVA`, `System Loading`, `CIG MW`, `Outage SG`, `SG 1 MW`, `SG 2 MW`, `SG 3 MW`.

Targets (20): `RoCoF Bus 1`–`RoCoF Bus 9`, `RoCoF Worst`, `Nadir Bus 1`–`Nadir Bus 9`, `Nadir Worst`.


## 3. Repository structure

Individual_Project_Dissertation/
├── README.md
├── requirements.txt
├── data/
│   └── raw/
│       └── Nine_bus_system_frequency_response_N_minus_one_disturbance.csv
├── src/
│   ├── preprocessing.py
│   ├── train_mlp.py
│   ├── train_rf.py
│   └── train_xgb.py
└── notebooks/
    ├── 01_preprocessing.ipynb
    ├── 02_train_mlp.ipynb
    ├── 03_train_rf.ipynb
    ├── 04_train_xgb.ipynb
    ├── 05_SHAP_Computation.ipynb
    ├── 06_Global_SHAP_Analysis.ipynb
    └── 07_SHAP_Local_Analysis.ipynb


`src/` and `notebooks/` are equivalent run paths. The notebooks are the primary route used in this project (Google Colab, high-RAM CPU runtime). The scripts in `src/` provide the same logic for headless local execution.

4. Installation

The project was developed in Python 3.10 on Google Colab using a high-RAM CPU runtime. No GPU is required at any stage.

To install dependencies locally:

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the eight libraries used by the pipeline at the exact versions of the Colab environment in which the final run was produced:

```
numpy==2.0.2
pandas==2.2.2
scikit-learn==1.6.1
xgboost==3.2.0
shap==0.51.0
joblib==1.5.3
matplotlib==3.10.0
seaborn==0.13.2
```

Jupyter is intentionally not pinned: the project is run in Google Colab, which provides its own Jupyter stack. For local execution, any recent `notebook` or `jupyterlab` install works.

Place the raw dataset at `data/raw/Nine_bus_system_frequency_response_N_minus_one_disturbance.csv`.

5. How to run the software

Run the seven notebooks in numerical order. Each notebook persists its outputs (processed data, trained models, SHAP values) so later stages do not need to retrain or recompute earlier ones.


`01_preprocessing.ipynb` | seconds |
`02_train_mlp.ipynb` | ≈ 18.2 h |
`03_train_rf.ipynb` | ≈ 14.3 h |
`04_train_xgb.ipynb` | ≈ 1.17 h |
`05_SHAP_Computation.ipynb` | KernelExplainer (MLP) dominates |
`06_Global_SHAP_Analysis.ipynb` | minutes |
`07_SHAP_Local_Analysis.ipynb` | minutes |


6. Technical details

**Dataset.** 10,394 simulated N-1 contingency scenarios on a modified IEEE 9-bus system, with 9 pre-fault inputs and 20 stability targets per row.

**Cleaning.** Two rows in which no frequency event occurred are removed by raw-row index (`8095` and `8311`); these were identified during dataset audit as zero-disturbance cases. A further 90 rows in which at least one bus reports RoCoF below `−1.0` Hz/s are removed. This is an Operating-Domain Definition (ODD) decision and not a physics-validity claim: the tail represents only 0.87 % of the cleaned dataset (90 of 10,302 rows) and is too sparse to support reliable supervised learning across all 20 targets. The cleaned dataset contains 10,302 rows.

**Split and scaling.** Stratified 70/30 train-test split on `Outage SG` with `random_state=42`, giving 7,211 training rows and 3,091 test rows. `StandardScaler` is fitted on the training inputs only and applied to both sets. Targets are deliberately left in physical units (Hz/s for RoCoF, Hz for nadir) so that error metrics and SHAP values remain dimensionally meaningful and directly comparable to operational thresholds.

**Models.** Random Forest, XGBoost, and a Multilayer Perceptron (MLP) are each trained per target via 5-fold `GridSearchCV` selecting on mean cross-validated R². The MLP uses a two-phase strategy: Phase 2 retrains on an expanded grid only when the Phase 1 cross-validated R² falls below 0.96. In the final run, Phase 2 was triggered for 7 of the 20 MLP targets.

**Evaluation.** Each model is scored on the held-out test set with R², RMSE, MAE, maximum absolute error, and maximum underestimate. Maximum underestimate is reported separately because under-prediction of RoCoF magnitude (or over-prediction of nadir) is the safety-relevant failure direction for this surrogate, and average error metrics can mask it. The train-test R² gap is checked against an overfit threshold.

**SHAP.** TreeExplainer (exact) is used for Random Forest and XGBoost. KernelExplainer (model-agnostic, approximate) is used for the MLP, with a background sample of approximately 500 rows drawn from the training set. SHAP values are computed on the full 3,091-row test set against the unscaled targets, so SHAP magnitudes are in Hz/s for RoCoF and Hz for nadir. SHAP values explain model behaviour and are not treated as evidence of causal physical mechanisms.

**Cross-algorithm consistency.** For each of the 20 targets, mean |SHAP| feature rankings are computed per algorithm. Spearman rank correlation is then computed between every pair of algorithms (MLP vs RF, MLP vs XGBoost, RF vs XGBoost), and the mean of the three pairwise correlations gives a single consistency score per target. A target is flagged as consistent when mean ρ ≥ 0.7. The threshold of 0.7 is fixed in the dissertation methodology and implemented in 06_Global_SHAP_Analysis.ipynb.


7. Known issues and future improvements

- KernelExplainer cost dominates the SHAP stage for the MLP. The 500-row background sample is a practical compromise on Colab; a larger background would tighten the SHAP estimate but at substantial additional runtime cost.
- MLP training time (≈ 18 h) is the bottleneck of the pipeline and is not parallelised across targets in the current implementation.
- The ODD filter (`RoCoF < −1.0` Hz/s) is a data-coverage decision based on the available simulation set, not a physics-based or protection-based criterion. Surrogate predictions outside the operating domain are not validated and should not be used
- The SHAP analysis explains the behaviour of the trained surrogate models. Treating SHAP attributions as confirmed physical mechanisms is outside the scope of this project.
- Future work: extension to larger network topologies (39-bus, 68-bus); replacement of the static feature set with a time-series representation suitable for a recurrent or temporal-convolutional surrogate; comparison with additional explainers (LIME, integrated gradients) to triangulate against SHAP.

#8. Third-party code, licences, and academic integrity

This project depends on the following open-source libraries, all installed via PyPI under their respective licences: NumPy (BSD-3), pandas (BSD-3), scikit-learn (BSD-3), XGBoost (Apache-2.0), SHAP (MIT), joblib (BSD-3), Matplotlib (Matplotlib licence, BSD-style), seaborn (BSD-3). No source code is reused verbatim from any external repository. Methodological reference is made to Kilembe et al. [1], Hamilton and Papadopoulos [2], and Lundberg and Lee [3]; no code from those works is incorporated.

Generative AI (Anthropic Claude) was used during this project for design-decision discussion and logic-checking. All code, results, interpretation, and dissertation text are my own work, verified against the source data and source notebooks. The candidate accepts full responsibility for the contents of this repository in line with the University of Manchester regulations on academic integrity.

9. Licence

Released under the MIT Licence.


## References

[1] N. Kilembe, R. Hamilton, and P. N. Papadopoulos, "Explainable machine learning for locational frequency stability assessment in low-inertia power systems," *International Journal of Electrical Power & Energy Systems*, vol. 170, 2025.

[2] R. Hamilton and P. N. Papadopoulos, "Using SHAP values and machine learning to understand trends in the transient stability limit," *IEEE Transactions on Power Systems*, 2023.

[3] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, 2017, pp. 4765–4774.
