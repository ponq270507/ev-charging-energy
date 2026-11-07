# Shared-input, equal-trial-budget sensitivity study

Six distinct configurations per tunable family, five original temporal folds, one fixed OLS reference: **185 real fits**. Each family includes its original matched-input setting. No final-test scoring or primary-model change.

| Model | Fixed CV R2 | Searched CV R2 | SD | MAE (kWh) | Selected trial |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.545866 | 0.546657 | 0.027681 | 6.424340 | 3 |
| CatBoost | 0.541030 | 0.544592 | 0.029678 | 6.492281 | 3 |
| HGB | 0.542229 | 0.543803 | 0.036401 | 6.473546 | 3 |
| LightGBM | 0.539383 | 0.543320 | 0.036083 | 6.482316 | 3 |
| Ridge | 0.477387 | 0.520741 | 0.027870 | 6.740518 | 4 |
| kNN | 0.473476 | 0.473476 | 0.038243 | 7.092024 | 1 |
| Linear Regression | 0.126788 | 0.126788 | 0.867204 | 8.716178 | 1 |

## Interpretation

Equal trial count does not mean equal wall-clock cost, exhaustive coverage, or equal historical research effort. Inputs and every train/validation row are identical within a fold; the encoder is fitted once on that fold's training rows. All fitting rows are used, including for kNN. Raw UserID is excluded; personal histories remain.

Selection and reporting use the same folds. The selected mean CV scores are exploratory sensitivity results, not nested estimates. Different family-specific spaces could change ranks. No universal best-algorithm claim follows.

Original pipeline rankings are retained as a separate historical result. Final test was already examined in the prior study; this extra search cannot make that test independent.

The locked protocol, all candidates and spaces, all 185 fold records, validation truths/predictions and primary artifact hashes are in this folder. `python verify_balanced_search.py` recomputes every one of the 740 fold metrics and checks the experiment structure.

## Selected settings

- Random Forest: trial 3; `{"max_depth": 16, "max_features": 0.65, "min_samples_leaf": 8, "n_estimators": 140}`
- CatBoost: trial 3; `{"depth": 7, "iterations": 400, "l2_leaf_reg": 30, "learning_rate": 0.06}`
- HGB: trial 3; `{"l2_regularization": 20, "learning_rate": 0.07, "max_iter": 180, "max_leaf_nodes": 15, "min_samples_leaf": 20}`
- LightGBM: trial 3; `{"learning_rate": 0.03, "min_child_samples": 120, "n_estimators": 450, "num_leaves": 15, "reg_lambda": 5}`
- Ridge: trial 4; `{"alpha": 1000}`
- kNN: trial 1; `{"n_neighbors": 25, "weights": "distance"}`
- Linear Regression: trial 1; `{}`
