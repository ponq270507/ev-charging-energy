# Retrospective nested temporal assessment

| model | outer_R2 | outer_R2_sd | outer_MAE | outer_RMSE | outer_folds |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 0.546578 | 0.027618 | 6.420736 | 9.198232 | 5 |
| Nested family selector | 0.544976 | 0.028731 | 6.459459 | 9.214174 | 5 |
| HGB | 0.544553 | 0.029416 | 6.497443 | 9.218400 | 5 |
| LightGBM | 0.543320 | 0.036083 | 6.482316 | 9.228920 | 5 |
| CatBoost | 0.542741 | 0.026979 | 6.511171 | 9.237125 | 5 |
| Ridge | 0.518698 | 0.029532 | 6.754541 | 9.476907 | 5 |
| kNN | 0.472039 | 0.040616 | 7.103690 | 9.923424 | 5 |
| User historical mean | 0.451397 | 0.031557 | 6.993421 | 10.118508 | 5 |
| Linear Regression | 0.126788 | 0.867204 | 8.716178 | 11.865873 | 5 |
| Dummy Mean | -0.001103 | 0.002078 | 10.610373 | 13.673336 | 5 |

595 actual fits: 555 inner, 35 outer regression, 5 Dummy. User historical mean is unfitted; selector reuses the inner-selected family predictions. All seven families share the same input matrix within each partition, with independent train-fitted encoders per partition. Inner labels are available before outer fitting; training labels are available before the relevant validation boundary.

`protocol.json` was recorded before fitting; `choices_before_outer_scoring.json` was recorded before outer fitting/scoring. Choices use only inner mean R2. See `verification.json` and the per-fit truth/prediction records.

This procedure excludes final-test scoring and leaves original artifacts unchanged. Candidate spaces and outer periods were previously explored, so this is a retrospective check. It does not confirm the original native-input blend selection or create an independent blind test. Equal trial counts do not equalize runtime or exhaustive search coverage.
