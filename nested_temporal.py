"""Retrospective nested temporal assessment on development dates only.

Five original outer folds; three expanding inner folds within each outer train.
The 37 pre-existing shared-input configurations are frozen, with six per tunable
family and one OLS reference. Every inner encoder is fitted on inner-train rows.
Inner validation labels must also be available before the outer fitting boundary.
555 inner + 35 outer regression fits + 5 Dummy Mean fits = 595 actual fits.
The original primary bundle/final test are not changed or rescored.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse, hashlib, json, tempfile, time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from threadpoolctl import threadpool_limits
from arrival_history import load_and_audit, build_features, CAT
from train_arrival import temporal_folds, metrics
from balanced_search import candidates, array_sha, sha

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'analysis_outputs/nested_temporal'
SEED=42


def utc():return datetime.now(timezone.utc).isoformat()


def design(d):
    cutoff=json.loads((ROOT/'analysis_outputs/protocol_before_test.json').read_text())['test_start']
    d=d[d.date_group.lt(cutoff)].copy().reset_index(drop=True)
    dev=d[d.label_available_at.lt(pd.Timestamp(cutoff))]
    outer,table=temporal_folds(d,dev)
    splits=[]
    for o,(oti,ovi) in enumerate(outer,1):
        boundary=d.StartDatetime.iloc[ovi].min().normalize()
        context=d[d.date_group.lt(str(boundary.date()))&d.label_available_at.lt(boundary)]
        dates=np.sort(context.date_group.unique())
        outer_mask=np.zeros(len(d),dtype=bool);outer_mask[oti]=True
        for i,(a,b) in enumerate(TimeSeriesSplit(n_splits=3).split(dates),1):
            ib=pd.Timestamp(dates[b[0]])
            ti=np.flatnonzero(outer_mask&d.date_group.isin(dates[a])&d.label_available_at.lt(ib))
            vi=np.flatnonzero(outer_mask&d.date_group.isin(dates[b]))
            assert len(ti)>50 and len(vi)>50
            assert set(ti).isdisjoint(vi) and set(ti)|set(vi)<=set(oti)
            assert d.label_available_at.iloc[ti].max()<d.StartDatetime.iloc[vi].min()
            assert d.label_available_at.iloc[vi].max()<boundary
            splits.append({'key':f'o{o}_i{i}','stage':'inner','outer':o,'inner':i,
                'train':ti,'valid':vi,'boundary':str(ib),'outer_boundary':str(boundary)})
        splits.append({'key':f'o{o}','stage':'outer','outer':o,'inner':0,
            'train':oti,'valid':ovi,'boundary':str(boundary),'outer_boundary':str(boundary)})
    return d,cutoff,table,splits


def prepare_cache(d,X,splits,cache):
    cats=[c for c in CAT if c in X];nums=[c for c in X if c not in cats]
    info=[]
    for spec in splits:
        key=spec['key'];ti=spec['train'];vi=spec['valid']
        pre=ColumnTransformer([
            ('num',Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]),nums),
            ('cat',OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency=10,max_categories=64,sparse_output=False),cats)])
        a=np.asarray(pre.fit_transform(X.iloc[ti]),dtype=np.float32)
        v=np.asarray(pre.transform(X.iloc[vi]),dtype=np.float32)
        assert np.isfinite(a).all() and np.isfinite(v).all()
        for name,value in [('train',a),('valid',v),('ytrain',d.Demand.iloc[ti].to_numpy()),('yvalid',d.Demand.iloc[vi].to_numpy())]:
            np.save(cache/f'{key}_{name}.npy',value)
        meta={k:val for k,val in spec.items() if k not in ['train','valid']}
        meta.update(fit_rows=len(ti),valid_rows=len(vi),encoded_columns=a.shape[1],
            train_matrix_sha256=array_sha(a),valid_matrix_sha256=array_sha(v),
            train_rows_sha256=array_sha(d.source_row.iloc[ti].to_numpy()),
            valid_rows_sha256=array_sha(d.source_row.iloc[vi].to_numpy()))
        info.append(meta)
        np.savez_compressed(OUT/'truth'/f'{key}.npz',y=d.Demand.iloc[vi].to_numpy(),
            source_row=d.source_row.iloc[vi].to_numpy(),fit_source_row=d.source_row.iloc[ti].to_numpy())
    return info


def fit_one(job,cache_path):
    m=job['split'];key=m['key'];cache=Path(cache_path)
    a=np.load(cache/f'{key}_train.npy',mmap_mode='r');v=np.load(cache/f'{key}_valid.npy',mmap_mode='r')
    y=np.load(cache/f'{key}_ytrain.npy',mmap_mode='r');yv=np.load(cache/f'{key}_yvalid.npy',mmap_mode='r')
    assert array_sha(a)==m['train_matrix_sha256'] and array_sha(v)==m['valid_matrix_sha256']
    f=job['family'];p=dict(job['params'])
    constructors={'linear':LinearRegression,'ridge':Ridge,'knn':KNeighborsRegressor,'rf':RandomForestRegressor,
        'hgb':HistGradientBoostingRegressor,'lgbm':LGBMRegressor,'catboost':CatBoostRegressor,'dummy':DummyRegressor}
    if f=='rf':p.update(n_jobs=2,random_state=SEED,max_samples=min(30000,len(y)))
    if f=='hgb':p.update(early_stopping=False,random_state=SEED)
    if f=='knn':p.update(n_jobs=2)
    if f=='lgbm':p.update(n_jobs=2,random_state=SEED,verbosity=-1)
    if f=='catboost':p.update(loss_function='RMSE',random_seed=SEED,thread_count=2,verbose=False,allow_writing_files=False)
    if f=='dummy':p.update(strategy='mean')
    model=constructors[f](**p);start=time.monotonic()
    with threadpool_limits(2):
        model.fit(a,y);pred=np.maximum(0,np.asarray(model.predict(v),dtype=float))
    assert pred.shape==yv.shape and np.isfinite(pred).all()
    row={**m,'job_id':job['job_id'],'model':job['model'],'family':f,'trial':job['trial'],
        'params':json.dumps(job['params'],sort_keys=True),'seconds':time.monotonic()-start,
        'selection_sha256':job.get('selection_sha256',''),**metrics(yv,pred)}
    return row,pred


def save_row(row,pred):
    name=row['job_id']
    np.savez_compressed(OUT/'predictions'/f'{name}.npz',prediction=pred)
    (OUT/'records'/f'{name}.json').write_text(json.dumps(row,indent=2))


def execute(jobs,cache,workers):
    pending=[]
    for j in jobs:
        path=OUT/'records'/f"{j['job_id']}.json"
        pp=OUT/'predictions'/f"{j['job_id']}.npz"
        if path.exists() and pp.exists():
            row=json.loads(path.read_text())
            assert row['params']==json.dumps(j['params'],sort_keys=True)
            assert row['train_matrix_sha256']==j['split']['train_matrix_sha256']
            assert row['selection_sha256']==j.get('selection_sha256','')
        else:pending.append(j)
    def cost(j):
        p=j['params'];n=j['split']['fit_rows'];f=j['family']
        if f=='rf':return n*p['n_estimators']*p['max_features']/max(1,p['min_samples_leaf'])
        if f=='catboost':return n*p['iterations']*2**p['depth']/200
        if f=='lgbm':return n*p['n_estimators']*p['num_leaves']/150
        if f=='hgb':return n*p['max_iter']*p['max_leaf_nodes']/150
        return n/10
    pending.sort(key=cost,reverse=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures=[executor.submit(fit_one,j,str(cache)) for j in pending]
        for k,future in enumerate(as_completed(futures),1):
            row,pred=future.result();save_row(row,pred)
            if k%15==0 or k==len(pending):
                print(f"{row['stage']}: {k}/{len(pending)} new fits; last {row['job_id']}",flush=True)


def choose_inner(configs):
    rows=[json.loads(p.read_text()) for p in sorted((OUT/'records').glob('o*_i*__*.json'))]
    f=pd.DataFrame(rows)
    assert len(f)==555
    summary=f.groupby(['outer','model','family','trial'],sort=False).agg(inner_R2=('R2','mean'),
        inner_MAE=('MAE','mean'),inner_folds=('inner','nunique')).reset_index()
    assert summary.inner_folds.eq(3).all()
    summary=summary.sort_values(['outer','model','inner_R2','trial'],ascending=[True,True,False,True])
    summary.to_csv(OUT/'inner_search.csv',index=False)
    best=summary.groupby(['outer','model'],sort=False).head(1)
    choices=[]
    for o,g in best.groupby('outer'):
        selected=g.sort_values(['inner_R2','model','trial'],ascending=[False,True,True]).iloc[0]
        for r in g.itertuples():
            spec=next(c for c in configs if c['model']==r.model and c['trial']==r.trial)
            choices.append({'outer':int(o),'model':r.model,'family':r.family,'trial':int(r.trial),
                'inner_R2':r.inner_R2,'inner_MAE':r.inner_MAE,'params':spec['params'],
                'selected_family':r.model==selected.model})
    return choices


def final_summaries(choices,d,X,splits):
    records=[json.loads(p.read_text()) for p in sorted((OUT/'records').glob('o*__*.json'))]
    outer=pd.DataFrame([r for r in records if r['stage']=='outer'])
    assert len(outer)==40
    extra=[]
    for spec in [s for s in splits if s['stage']=='outer']:
        o=spec['outer'];vi=spec['valid'];xx=X.iloc[vi]
        pred=xx.user_energy_mean.where(xx.user_log_count.gt(0),xx.global_energy_mean).to_numpy(float)
        assert np.isfinite(pred).all()
        key=f'o{o}__user_mean'
        np.savez_compressed(OUT/'predictions'/f'{key}.npz',prediction=pred)
        extra.append({'stage':'outer','outer':o,'model':'User historical mean','family':'user_mean','trial':0,
            'job_id':key,'fit_rows':len(spec['train']),'valid_rows':len(vi),'fit_performed':False,
            **metrics(d.Demand.iloc[vi],pred)})
        winner=next(c for c in choices if c['outer']==o and c['selected_family'])
        row=outer[outer.outer.eq(o)&outer.model.eq(winner['model'])].iloc[0].to_dict()
        row.update(model='Nested family selector',selected_family=winner['model'],fit_performed=False)
        extra.append(row)
    outer=pd.concat([outer,pd.DataFrame(extra)],ignore_index=True)
    outer.to_csv(OUT/'outer_metrics.csv',index=False)
    summary=outer.groupby('model',sort=False).agg(outer_R2=('R2','mean'),outer_R2_sd=('R2','std'),
        outer_MAE=('MAE','mean'),outer_RMSE=('RMSE','mean'),outer_folds=('outer','nunique')).reset_index()
    assert summary.outer_folds.eq(5).all()
    summary.sort_values(['outer_R2','model'],ascending=[False,True]).to_csv(OUT/'comparison.csv',index=False)
    print(summary.to_string(index=False),flush=True)


def run(workers=4,resume=False):
    for folder in [OUT,OUT/'truth',OUT/'records',OUT/'predictions']:folder.mkdir(parents=True,exist_ok=True)
    source=ROOT/'data/ChargingRecords_original.csv'
    configs,spaces=candidates()
    old=json.loads((ROOT/'analysis_outputs/balanced_search/protocol.json').read_text())
    assert configs==old['candidates'] and sha(source)==old['source_sha256']
    if (OUT/'protocol.json').exists() and not resume:
        raise RuntimeError('Nested run already exists. Use --resume for the identical locked protocol, or preserve/rename it before rerunning.')
    d,_,_=load_and_audit(source);d,cutoff,fold_table,splits=design(d)
    X,_=build_features(d,d);X=X.drop(columns='UserID');assert len(X.columns)==136
    hashes={n:sha(ROOT/n) for n in ['nested_temporal.py','balanced_search.py','arrival_history.py','train_arrival.py']}
    with tempfile.TemporaryDirectory(prefix='group25_nested_') as folder:
        cache=Path(folder);info=prepare_cache(d,X,splits,cache)
        protocol={'created_utc':utc(),'purpose':'retrospective nested temporal model-family assessment on pre-test dates',
            'outer_folds':5,'inner_folds_per_outer':3,'candidates':configs,'spaces':spaces,
            'expected_inner_fits':555,'expected_outer_regression_fits':35,'expected_dummy_fits':5,'expected_total_fits':595,
            'selector':'Within each outer fold, choose each family configuration by highest unweighted mean inner R2; choose the family by that inner score only. Exact ties use model name then trial.',
            'common_raw_predictors':list(X.columns),'raw_predictors':136,'seed':SEED,
            'encoding':'Each inner/outer training partition independently fits numeric median+scaling and one-hot min_frequency10/max_categories64; shared by all candidates in that partition.',
            'inner_label_availability':'Both inner fit and inner validation rows are restricted to the outer fitting set. Inner fit labels must be available before inner validation boundary.',
            'history':'online completion-gated replay; EndDatetime+60seconds < arrival; no model updates within a validation window',
            'final_test_start':cutoff,'final_test_scored':False,'primary_model_changed':False,
            'source_sha256':sha(source),'code_sha256':hashes,'original_artifact_hashes':old['original_artifact_hashes'],
            'partitions':info,'limits':['Retrospective: outer periods and dataset were previously inspected in earlier studies.',
                'Frozen candidate spaces reflect prior development; this does not create an independent blind test.',
                'Validates the shared-input seven-family selection procedure, not the old native-input CatBoost blend selection procedure.',
                'Finite equal trial counts do not mean equal compute or exhaustive search.']}
        if (OUT/'protocol.json').exists():
            saved=json.loads((OUT/'protocol.json').read_text())
            for key in protocol:
                if key!='created_utc':assert protocol[key]==saved[key],key
        else:(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
        fold_table.to_csv(OUT/'outer_folds.csv',index=False)
        inner_jobs=[]
        for m in [m for m in info if m['stage']=='inner']:
            for c in configs:inner_jobs.append({**c,'split':m,'job_id':m['key']+f"__{c['family']}_{c['trial']}"})
        print('LOCKED NESTED PROTOCOL: 555 inner + 35 outer regression + 5 Dummy fits; final test excluded.',flush=True)
        execute(inner_jobs,cache,workers)
        choices=choose_inner(configs)
        selection={'created_utc':utc(),'outer_scores_used_for_selection':False,'choices':choices}
        choice_path=OUT/'choices_before_outer_scoring.json'
        if choice_path.exists():
            assert json.loads(choice_path.read_text())['choices']==choices
        else:choice_path.write_text(json.dumps(selection,indent=2))
        choice_sha=sha(choice_path)
        outer_jobs=[]
        for c in choices:
            m=next(m for m in info if m['stage']=='outer' and m['outer']==c['outer'])
            outer_jobs.append({**c,'split':m,'selection_sha256':choice_sha,
                'job_id':m['key']+f"__{c['family']}_{c['trial']}"})
        for m in [m for m in info if m['stage']=='outer']:
            outer_jobs.append({'model':'Dummy Mean','family':'dummy','trial':0,'params':{},'split':m,
                'selection_sha256':choice_sha,'job_id':m['key']+'__dummy_0'})
        print('INNER CHOICES LOCKED. Starting 40 outer fits.',flush=True)
        execute(outer_jobs,cache,workers)
        final_summaries(choices,d,X,splits)
    records=[json.loads(p.read_text()) for p in (OUT/'records').glob('*.json')]
    assert len(records)==595
    pd.DataFrame(records).sort_values(['stage','outer','inner','model','trial']).to_csv(OUT/'all_fit_metrics.csv',index=False)
    for rel,before in old['original_artifact_hashes'].items():assert sha(ROOT/rel)==before,rel
    completed={'completed_utc':utc(),'actual_fits':595,'protocol_sha256':sha(OUT/'protocol.json'),
        'selection_sha256':sha(OUT/'choices_before_outer_scoring.json'),
        'all_fit_metrics_sha256':sha(OUT/'all_fit_metrics.csv'),'primary_model_unchanged':True,'independent_test_created':False}
    (OUT/'completed.json').write_text(json.dumps(completed,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if not 1<=a.workers<=4:p.error('Use 1--4 workers (two threads per fit).')
    run(a.workers,a.resume)
