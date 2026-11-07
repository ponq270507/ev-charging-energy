"""Development-only shared-input, equal-trial-budget sensitivity study.

Six distinct configurations for each tunable family, five original temporal
folds, and one fixed ordinary least-squares reference: 185 actual fits.
No final-test predictions are read or made. Original primary selection is kept.
Run from the package root: python balanced_search.py --workers 3
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import hashlib
import json
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from threadpoolctl import threadpool_limits

from arrival_history import load_and_audit, build_features, CAT
from train_arrival import base_specs, temporal_folds, metrics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'analysis_outputs/balanced_search'
SEED = 42
TRIALS = 6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.shape).encode() + str(a.dtype).encode() + a.tobytes()).hexdigest()


def candidates():
    """Fix the complete candidate list before observing any new fit scores.

    Every tunable family includes its original matched-input setting plus five
    unique settings from the declared space. Ridge and kNN use all six settings.
    OLS is an untuned reference; duplicating its identical setting is wasteful.
    """
    spaces = {
        'Ridge': {'alpha': [.1, 1, 10, 100, 1000, 10000]},
        'kNN': {'n_neighbors': [15, 25, 50], 'weights': ['uniform', 'distance']},
        'Random Forest': {'n_estimators': [100, 140], 'max_depth': [10, 16],
            'min_samples_leaf': [5, 8, 20], 'max_features': [.65, .85]},
        'HGB': {'max_iter': [180, 260, 400], 'max_leaf_nodes': [7, 15, 31],
            'min_samples_leaf': [20, 30, 60], 'l2_regularization': [0, 20, 50],
            'learning_rate': [.03, .05, .07]},
        'LightGBM': {'n_estimators': [450, 700, 1000], 'num_leaves': [7, 15, 31],
            'min_child_samples': [25, 60, 120], 'learning_rate': [.025, .03, .07],
            'reg_lambda': [5, 20, 50]},
        'CatBoost': {'iterations': [400, 500, 700], 'depth': [5, 6, 7],
            'learning_rate': [.035, .05, .06], 'l2_leaf_reg': [10, 20, 30]},
    }
    specs = base_specs()
    jobs = []
    for name, space in spaces.items():
        params = [dict(specs[name]['params'])]
        # A shuffled finite grid gives deterministic sampling without replacement.
        grid = list(ParameterGrid(space))
        for index in np.random.default_rng(SEED).permutation(len(grid)):
            item = dict(grid[int(index)])
            if item not in params:
                params.append(item)
            if len(params) == TRIALS:
                break
        assert len(params) == TRIALS
        assert len({json.dumps(p, sort_keys=True) for p in params}) == TRIALS
        for trial, params_i in enumerate(params, 1):
            jobs.append({'model': name, 'family': specs[name]['family'],
                'trial': trial, 'params': params_i})
    jobs.append({'model': 'Linear Regression', 'family': 'linear', 'trial': 1, 'params': {}})
    return jobs, spaces


def fit_candidate(job, cache_path, fold_info):
    constructors = {'linear': LinearRegression, 'ridge': Ridge,
        'knn': KNeighborsRegressor, 'rf': RandomForestRegressor,
        'hgb': HistGradientBoostingRegressor, 'lgbm': LGBMRegressor,
        'catboost': CatBoostRegressor}
    cache = Path(cache_path)
    rows, predictions = [], []
    for info in fold_info:
        fold = info['fold']
        a = np.load(cache / f'{fold}_train.npy', mmap_mode='r')
        v = np.load(cache / f'{fold}_valid.npy', mmap_mode='r')
        y = np.load(cache / f'{fold}_ytrain.npy', mmap_mode='r')
        yv = np.load(cache / f'{fold}_yvalid.npy', mmap_mode='r')
        assert array_sha(a) == info['train_matrix_sha256']
        assert array_sha(v) == info['valid_matrix_sha256']
        params = dict(job['params'])
        family = job['family']
        if family == 'rf':
            params.update(n_jobs=2, random_state=SEED, max_samples=min(30000, len(a)))
        if family == 'hgb':
            params.update(early_stopping=False, random_state=SEED)
        if family == 'knn':
            params.update(n_jobs=2)
        if family == 'lgbm':
            params.update(n_jobs=2, random_state=SEED, verbosity=-1)
        if family == 'catboost':
            params.update(loss_function='RMSE', random_seed=SEED, thread_count=2,
                verbose=False, allow_writing_files=False)
        model = constructors[family](**params)
        start = time.monotonic()
        with threadpool_limits(limits=2):
            model.fit(a, y)
            pred = np.maximum(0, np.asarray(model.predict(v), dtype=float))
        assert pred.shape == yv.shape and np.isfinite(pred).all()
        predictions.append(pred)
        rows.append({'model': job['model'], 'family': family, 'trial': job['trial'],
            'fold': fold, 'fit_rows': len(a), 'valid_rows': len(v),
            'encoded_columns': a.shape[1], 'train_matrix_sha256': info['train_matrix_sha256'],
            'valid_matrix_sha256': info['valid_matrix_sha256'],
            'train_rows_sha256': info['train_rows_sha256'],
            'valid_rows_sha256': info['valid_rows_sha256'],
            'seconds': time.monotonic() - start, **metrics(yv, pred)})
    return job, rows, np.concatenate(predictions)


def summarize(rows):
    frame = pd.DataFrame(rows)
    trials = frame.groupby(['model', 'family', 'trial'], sort=False).agg(
        cv_R2=('R2', 'mean'), cv_R2_sd=('R2', 'std'), cv_MAE=('MAE', 'mean'),
        cv_RMSE=('RMSE', 'mean'), total_seconds=('seconds', 'sum')).reset_index()
    trials = trials.sort_values(['model', 'cv_R2', 'trial'], ascending=[True, False, True])
    trials.to_csv(OUT / 'trials.csv', index=False)
    summary = trials.groupby('model', sort=False).head(1).copy()
    summary['configurations'] = summary['model'].map(trials.groupby('model').size())
    fixed = frame[frame.trial.eq(1)].groupby('model').R2.mean()
    summary['fixed_cv_R2'] = summary['model'].map(fixed)
    summary['cv_R2_gain'] = summary.cv_R2 - summary.fixed_cv_R2
    summary = summary.sort_values(['cv_R2', 'model'], ascending=[False, True])
    summary.to_csv(OUT / 'comparison.csv', index=False)
    return summary


def run(workers=3):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'predictions').mkdir(exist_ok=True)
    if (OUT / 'protocol.json').exists():
        raise RuntimeError('Existing balanced-search run found; keep it as evidence or move it before a fresh run.')
    jobs, spaces = candidates()
    source = ROOT / 'data/ChargingRecords_original.csv'
    locked = json.loads((ROOT / 'analysis_outputs/protocol_before_test.json').read_text())
    cutoff = locked['test_start']
    d, _, _ = load_and_audit(source)
    # No final-test row enters feature construction or scoring for this search.
    d = d[d.date_group.lt(cutoff)].copy().reset_index(drop=True)
    dev = d[d.label_available_at.lt(pd.Timestamp(cutoff))].copy()
    folds, fold_table = temporal_folds(d, dev)
    X, _ = build_features(d, d)
    X = X.drop(columns='UserID')
    assert X.shape[1] == 136
    cats = [c for c in CAT if c in X]
    nums = [c for c in X if c not in cats]
    original_files = [ROOT / 'analysis_outputs/models/selection_before_test.json',
        ROOT / 'analysis_outputs/models/arrival_model_bundle.joblib',
        ROOT / 'analysis_outputs/models/personalized_test_predictions.csv']
    protected = {str(p.relative_to(ROOT)): sha(p) for p in original_files}
    protocol = {'started_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'post-review development-only equal-trial-budget sensitivity study',
        'seed': SEED, 'raw_predictors': list(X.columns), 'raw_predictor_count': len(X.columns),
        'excluded': 'raw UserID only; all personal histories retained',
        'input': 'common fold-fitted median imputation, scaling, and dense one-hot encoding',
        'onehot_min_frequency': 10, 'onehot_max_categories': 64,
        'candidates_per_tunable_family': TRIALS, 'tunable_families': 6,
        'untuned_reference': 'Linear Regression; single OLS configuration',
        'number_folds': 5, 'expected_fits': 185,
        'budget_unit': 'distinct configurations times the same five folds; not equal wall-clock time',
        'sampling': 'original setting plus first five distinct settings from a seed-42 random permutation of the full finite grid',
        'search_spaces': spaces, 'candidates': jobs,
        'selection': 'highest unweighted mean fold R2 per family; lower trial breaks exact ties',
        'search_scores_are_nested': False,
        'knn_training_cap': None, 'same_training_rows': True,
        'final_test_start': cutoff, 'last_search_arrival': str(d.StartDatetime.max()),
        'final_test_scored': False, 'primary_selection_changed': False,
        'source_sha256': sha(source), 'code_sha256': sha(__file__),
        'original_artifact_hashes': protected,
        'cautions': ['finite, family-specific spaces', 'same-fold selection scores are exploratory',
            'prior test exposure persists', 'shared encoding may disadvantage native-category models']}
    fold_table.to_csv(OUT / 'folds.csv', index=False)
    rows = []
    with tempfile.TemporaryDirectory(prefix='group25_balanced_') as cache_path:
        cache = Path(cache_path)
        fold_info, truth, ids, fold_ids = [], [], [], []
        for fold, (ti, vi) in enumerate(folds, 1):
            pre = ColumnTransformer([
                ('num', Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                    ('scale', StandardScaler())]), nums),
                ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=10,
                    max_categories=64, sparse_output=False), cats)])
            a = np.asarray(pre.fit_transform(X.iloc[ti]), dtype=np.float32)
            v = np.asarray(pre.transform(X.iloc[vi]), dtype=np.float32)
            assert np.isfinite(a).all() and np.isfinite(v).all()
            for label, values in [('train', a), ('valid', v),
                    ('ytrain', d.Demand.iloc[ti].to_numpy()), ('yvalid', d.Demand.iloc[vi].to_numpy())]:
                np.save(cache / f'{fold}_{label}.npy', values)
            info = {'fold': fold, 'fit_rows': len(ti), 'valid_rows': len(vi),
                'encoded_columns': a.shape[1], 'train_matrix_sha256': array_sha(a),
                'valid_matrix_sha256': array_sha(v),
                'train_rows_sha256': array_sha(d.source_row.iloc[ti].to_numpy()),
                'valid_rows_sha256': array_sha(d.source_row.iloc[vi].to_numpy())}
            fold_info.append(info)
            truth.append(d.Demand.iloc[vi].to_numpy())
            ids.append(d.source_row.iloc[vi].to_numpy())
            fold_ids.append(np.full(len(vi), fold, dtype=np.int8))
        protocol['fold_input_hashes'] = fold_info
        # Written before the first model fit. Do not edit this record after seeing scores.
        (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2))
        np.savez_compressed(OUT / 'validation_truth.npz', y=np.concatenate(truth),
            source_row=np.concatenate(ids), fold=np.concatenate(fold_ids))
        print('LOCKED PROTOCOL:', len(jobs), 'configurations; 185 fits; no final-test scoring.', flush=True)
        # Expensive jobs first; each independent worker is limited to two threads.
        order = {'rf': 0, 'catboost': 1, 'lgbm': 2, 'hgb': 3, 'knn': 4, 'ridge': 5, 'linear': 6}
        jobs = sorted(jobs, key=lambda j: (order[j['family']], j['trial']))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(fit_candidate, j, cache_path, fold_info) for j in jobs]
            for done, future in enumerate(as_completed(futures), 1):
                job, records, pred = future.result()
                rows.extend(records)
                np.savez_compressed(OUT / 'predictions' / f"{job['family']}_{job['trial']}.npz", prediction=pred)
                pd.DataFrame(rows).sort_values(['model', 'trial', 'fold']).to_csv(OUT / 'fold_metrics.csv', index=False)
                score = float(np.mean([r['R2'] for r in records]))
                print(f"[{done}/{len(jobs)}] {job['model']} trial {job['trial']}: CV R2={score:.6f}", flush=True)
    assert len(rows) == 185
    for name, before in protected.items():
        assert sha(ROOT / name) == before, name
    summary = summarize(rows)
    complete = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'fit_count': len(rows),
        'primary_artifacts_unchanged': True, 'protocol_sha256': sha(OUT / 'protocol.json'),
        'fit_metrics_sha256': sha(OUT / 'fold_metrics.csv'), 'test_evaluation_performed': False}
    (OUT / 'completed.json').write_text(json.dumps(complete, indent=2))
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error('--workers must be between 1 and 4')
    run(args.workers)
