"""Review-response diagnostics. Fixed models; no change to primary selection.

Run from the package root: python feedback_checks.py
All seven families receive the identical fold-fitted numeric matrix and rows.
Unlike the original kNN pipeline, this controlled comparison has no 12k row cap.
"""
from pathlib import Path
import json, hashlib, time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from threadpoolctl import threadpool_limits
from arrival_history import load_and_audit, build_features, chronological_split, CAT
from train_arrival import base_specs, temporal_folds, metrics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'analysis_outputs/feedback'

def decision_checks():
    p = pd.read_csv(ROOT/'analysis_outputs/models/personalized_test_predictions.csv')
    b = pd.read_csv(ROOT/'analysis_outputs/revision/user_mean_predictions.csv')
    assert np.array_equal(p.source_row, b.source_row)
    selected = json.loads((ROOT/'analysis_outputs/manifest.json').read_text())['selected_model']
    y = p.Demand.to_numpy(); n=len(y); k=int(np.ceil(.2*n))
    # Exact-k definition and deterministic source-row tie-break for both rankings.
    true_top=set(np.lexsort((p.source_row.to_numpy(),-y))[:k])
    rows=[]
    for name, pred in [(selected,p[selected].to_numpy()),('User historical mean',b['User historical mean'].to_numpy())]:
        top=set(np.lexsort((p.source_row.to_numpy(),-pred))[:k])
        tail=y>40
        rows.append({'model':name,'n':n,'k':k,'top20_hits':len(top&true_top),
            'precision_at_20pct':len(top&true_top)/k,'high_energy_n':int(tail.sum()),
            'high_energy_MAE':float(np.abs(y[tail]-pred[tail]).mean()),
            'high_energy_bias_pred_minus_actual':float((pred[tail]-y[tail]).mean()),
            'high_energy_underpredicted_n':int((pred[tail]<y[tail]).sum()),
            'high_energy_below_half_actual_n':int((pred[tail]<.5*y[tail]).sum()),
            'low_energy_n':int((~tail).sum()),'low_energy_MAE':float(np.abs(y[~tail]-pred[~tail]).mean())})
    pd.DataFrame(rows).to_csv(OUT/'decision_and_tail.csv',index=False)
    return rows

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'data/ChargingRecords_original.csv'
    protocol={'raw_predictors':136,'excluded':'raw UserID only; personal histories retained',
        'encoding':'common numeric median imputation and standardization; common dense one-hot of all four categories',
        'onehot_min_frequency':10,'onehot_max_categories':64,'same_training_rows_for_all':True,
        'knn_training_cap':None,'folds':5,'selection_changed':False,'seed':42,
        'purpose':'post-review development-only controlled representation comparison; no test selection',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'matched_protocol.json').write_text(json.dumps(protocol,indent=2))
    print('Decision diagnostics:',decision_checks(),flush=True)
    d,_,_=load_and_audit(source)
    dev,_,_,_=chronological_split(d)
    folds,records=temporal_folds(d,dev)
    X,_=build_features(d,d); X=X.drop(columns='UserID')
    assert X.shape[1]==136
    cats=[c for c in CAT if c in X]; nums=[c for c in X if c not in cats]
    rows=[]; specs=base_specs()
    constructors={'linear':LinearRegression,'ridge':Ridge,'knn':KNeighborsRegressor,
        'rf':RandomForestRegressor,'hgb':HistGradientBoostingRegressor,'lgbm':LGBMRegressor,'catboost':CatBoostRegressor}
    for fold,(ti,vi) in enumerate(folds,1):
        pre=ColumnTransformer([
            ('num',Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]),nums),
            ('cat',OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency=10,max_categories=64,sparse_output=False),cats)])
        a=np.asarray(pre.fit_transform(X.iloc[ti]),dtype=np.float32)
        v=np.asarray(pre.transform(X.iloc[vi]),dtype=np.float32)
        assert np.isfinite(a).all() and np.isfinite(v).all()
        for name,spec in specs.items():
            family=spec['family']
            if family=='dummy':continue
            params=dict(spec['params'])
            if family=='rf':params.update(n_jobs=2,random_state=42,max_samples=min(30000,len(ti)))
            if family=='hgb':params.update(early_stopping=False,random_state=42)
            if family=='knn':params.update(n_jobs=2)
            if family=='lgbm':params.update(n_jobs=2,random_state=42,verbosity=-1)
            if family=='catboost':params.update(loss_function='RMSE',random_seed=42,thread_count=2,verbose=False,allow_writing_files=False)
            model=constructors[family](**params); start=time.time()
            with threadpool_limits(2):
                model.fit(a,d.Demand.iloc[ti].to_numpy())
                pred=np.maximum(0,np.asarray(model.predict(v)))
            row={'model':name,'fold':fold,'fit_rows':len(ti),'valid_rows':len(vi),
                 'encoded_columns':a.shape[1],'seconds':time.time()-start,**metrics(d.Demand.iloc[vi],pred)}
            rows.append(row);pd.DataFrame(rows).to_csv(OUT/'matched_folds.csv',index=False)
            print(row,flush=True)
    f=pd.DataFrame(rows)
    assert len(f)==35 and f.groupby('model').size().eq(5).all()
    summary=f.groupby('model',sort=False).agg(cv_R2=('R2','mean'),cv_R2_sd=('R2','std'),cv_MAE=('MAE','mean'),cv_RMSE=('RMSE','mean')).reset_index().sort_values('cv_R2',ascending=False)
    summary.to_csv(OUT/'matched_comparison.csv',index=False)
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':run()
