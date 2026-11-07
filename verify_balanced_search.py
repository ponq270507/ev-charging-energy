"""Recompute every balanced-search metric from saved validation predictions.

This verifies the completed run without refitting models or scoring final test.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from balanced_search import ROOT, OUT, candidates, sha
from train_arrival import metrics, temporal_folds
from arrival_history import load_and_audit


def verify():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    complete = json.loads((OUT / 'completed.json').read_text())
    assert protocol['code_sha256'] == sha(ROOT / 'balanced_search.py')
    assert protocol['source_sha256'] == sha(ROOT / 'data/ChargingRecords_original.csv')
    assert complete['protocol_sha256'] == sha(OUT / 'protocol.json')
    assert complete['fit_metrics_sha256'] == sha(OUT / 'fold_metrics.csv')
    jobs, spaces = candidates()
    assert protocol['candidates'] == jobs and protocol['search_spaces'] == spaces
    assert protocol['expected_fits'] == complete['fit_count'] == 185
    assert protocol['final_test_scored'] is complete['test_evaluation_performed'] is False
    assert protocol['primary_selection_changed'] is False
    assert protocol['last_search_arrival'] < protocol['final_test_start']
    for rel, before in protocol['original_artifact_hashes'].items():
        assert sha(ROOT / rel) == before, rel

    d, _, _ = load_and_audit(ROOT / 'data/ChargingRecords_original.csv')
    cutoff = protocol['final_test_start']
    d = d[d.date_group.lt(cutoff)].copy().reset_index(drop=True)
    dev = d[d.label_available_at.lt(pd.Timestamp(cutoff))].copy()
    folds, fold_table = temporal_folds(d, dev)
    pd.testing.assert_frame_equal(pd.read_csv(OUT / 'folds.csv'), fold_table, check_dtype=False)
    truth = np.load(OUT / 'validation_truth.npz')
    assert np.array_equal(truth['source_row'], np.concatenate([d.source_row.iloc[vi] for _, vi in folds]))
    assert np.array_equal(truth['y'], np.concatenate([d.Demand.iloc[vi] for _, vi in folds]))
    assert np.array_equal(truth['fold'], np.concatenate([np.full(len(vi), i) for i, (_, vi) in enumerate(folds, 1)]))
    assert len(set(truth['source_row'])) == len(truth['source_row'])

    frame = pd.read_csv(OUT / 'fold_metrics.csv')
    assert len(frame) == 185
    assert not frame.duplicated(['model', 'trial', 'fold']).any()
    assert frame.groupby(['model', 'trial']).size().eq(5).all()
    counts = frame.groupby('model').trial.nunique()
    assert counts.loc['Linear Regression'] == 1
    assert counts.drop('Linear Regression').eq(6).all() and len(counts) == 7
    metric_count = 0
    for job in jobs:
        pred = np.load(OUT / 'predictions' / f"{job['family']}_{job['trial']}.npz")['prediction']
        assert pred.shape == truth['y'].shape and np.isfinite(pred).all() and (pred >= 0).all()
        g = frame[frame.model.eq(job['model']) & frame.trial.eq(job['trial'])].set_index('fold')
        for fold in range(1, 6):
            mask = truth['fold'] == fold
            for key, value in metrics(truth['y'][mask], pred[mask]).items():
                assert np.isclose(value, g.loc[fold, key], rtol=1e-12, atol=1e-12), (job, fold, key)
                metric_count += 1

    for fold, g in frame.groupby('fold'):
        info = protocol['fold_input_hashes'][fold-1]
        for col in ['fit_rows', 'valid_rows', 'encoded_columns', 'train_matrix_sha256',
                'valid_matrix_sha256', 'train_rows_sha256', 'valid_rows_sha256']:
            assert g[col].nunique() == 1 and g[col].iloc[0] == info[col], (fold, col)
        assert g.fit_rows.iloc[0] == [3078, 6418, 11737, 17079, 23034][fold-1]
        assert g.valid_rows.iloc[0] == [3348, 5323, 5333, 5969, 6730][fold-1]

    trials = pd.read_csv(OUT / 'trials.csv').set_index(['model', 'trial'])
    summary = pd.read_csv(OUT / 'comparison.csv').set_index('model')
    for (name, trial), g in frame.groupby(['model', 'trial']):
        for col, value in {'cv_R2': g.R2.mean(), 'cv_R2_sd': g.R2.std(ddof=1),
                'cv_MAE': g.MAE.mean(), 'cv_RMSE': g.RMSE.mean()}.items():
            assert np.isclose(value, trials.loc[(name, trial), col], atol=1e-12)
    for name in counts.index:
        group = trials.loc[name].reset_index().sort_values(['cv_R2', 'trial'], ascending=[False, True])
        best = group.iloc[0]
        assert summary.loc[name, 'trial'] == best.trial
        for col in ['cv_R2', 'cv_R2_sd', 'cv_MAE', 'cv_RMSE']:
            assert np.isclose(summary.loc[name, col], best[col], atol=1e-12)

    old = pd.read_csv(ROOT / 'analysis_outputs/feedback/matched_comparison.csv').set_index('model')
    for name in counts.index:
        # Trial 1 is a fresh repeat of the old fixed matched-input comparison.
        for col in ['cv_R2', 'cv_MAE', 'cv_RMSE']:
            assert np.isclose(trials.loc[(name, 1), col], old.loc[name, col], atol=2e-6), (name, col)
    result = {'passed': True, 'actual_model_fits': 185, 'validation_metrics_recomputed': metric_count,
        'six_distinct_configurations_per_tunable_family': True,
        'seven_families_share_rows_and_matrices_within_each_fold': True,
        'original_fixed_results_reproduced': True, 'primary_artifacts_unchanged': True,
        'final_test_scored_by_new_search': False, 'independent_validation_performed': False,
        'selected_CV_scores_are_exploratory_not_nested_estimates': True}
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    verify()
