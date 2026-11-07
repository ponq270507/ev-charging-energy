# Reproduction protocol

## Environment

Python 3.12, CPU. Create a fresh virtual environment and run:

```sh
python -m pip install -r requirements.txt
python compare_uploaded_sources.py
python verify_results.py
python verify_feedback.py
python verify_balanced_search.py
python verify_nested_temporal.py
python verify_inference_ids.py
python verify_inference_inputs.py
python report_source/render_review_figures.py
```

Run commands from this project's ADY_Group25 directory. Execute the notebook in order
for SQL analysis, feature construction, diagnostic tables and cached experiment checks.
The setup cell installs pinned dependencies automatically in Colab. Restart the runtime
if a previously imported library was replaced. Default execution uses saved models.

## Data and prediction boundary

72,856 original sessions; 3,286 quarantined; 69,570 retained. The supplied Excel file
is compared cellwise after date/time normalization with the unchanged canonical CSV.
No additional external dataset is merged. Target: final session Demand in kWh, predicted
at arrival. A completed prior session contributes history only when its end time plus
the assumed 60-second reporting delay is strictly before the query arrival. Current
session duration, end time and final energy are unavailable as predictors.

Temporal cutoff: 2022-07-19. Development/test/purge: 52,229/17,310/31; identified-user
subsets: 29,775/10,999/16. 137 pre-encoding predictors. Primary equal CatBoost blend
is the historical CV-selected model. Online completed-label feedback updates history,
not fitted weights. The inference example without a completed feed uses frozen history.

## Expected checkpoints

Identified-user online test (10,999 sessions): R2 0.5111153964; MAE 6.8428568854 kWh;
RMSE 9.7993804832 kWh; MedAE 4.5308631872 kWh. The nested temporal selector's outer
mean R2 is approximately 0.5449763065. These are different evaluation protocols.
Saved fold predictions, partitions, selection and protocol/module hashes are checked.

## Optional full refitting

RUN_TRAIN: original model study (60 tuning fits); RUN_REVISION_CHECKS: diagnostic reruns;
RUN_FEEDBACK_CHECKS: 35 matched fits; RUN_BALANCED_SEARCH: 185 fits;
RUN_NESTED_TEMPORAL: 595 fits. Enable only after reading the corresponding notebook cells.
Back up the project before refitting because these modes overwrite saved outputs.
The included core estimator modules remain byte-identical to those recorded by historical
experiment protocols. Historical hashes are not rewritten to conceal changes.

## Scope and provenance

The new notebook is based on the newly supplied standalone notebook, with balanced/nested
cells and a repaired loader, requirements and figure export. The trained bundle and cached
experiments are inherited, explicitly identified in PACKAGE_INFO.json; no new training is
claimed. Historical model selection/test exposure remains a limitation. Current manuscript
text, numbering and citations must be reconciled with the separately maintained Overleaf source.
