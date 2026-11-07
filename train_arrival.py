"""Personalized arrival prediction with a causal, replayed feature store.

CV and test use online history: a label becomes a history event only after
EndDatetime + 60 seconds. Models are never refit using validation/test labels.
The frozen-history test is a separate diagnostic, using the same fitted model.
"""
from pathlib import Path
import json,time,platform
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import TimeSeriesSplit,ParameterSampler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,median_absolute_error
from threadpoolctl import threadpool_limits
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from arrival_history import CAT,CALENDAR,HISTORY,build_features,chronological_split,audit_feature_causality

SEED=42

def metrics(y,p):
    return {'R2':float(r2_score(y,p)),'MAE':float(mean_absolute_error(y,p)),
        'RMSE':float(np.sqrt(mean_squared_error(y,p))),'MedAE':float(median_absolute_error(y,p))}


def base_specs():
    return {
        'Dummy Mean':{'family':'dummy','params':{}},
        'Linear Regression':{'family':'linear','params':{}},
        'Ridge':{'family':'ridge','params':{'alpha':100}},
        'kNN':{'family':'knn','params':{'n_neighbors':25,'weights':'distance'},'training_cap':12000},
        'Random Forest':{'family':'rf','params':{'n_estimators':140,'max_depth':16,'min_samples_leaf':8,'max_features':.85}},
        'HGB':{'family':'hgb','params':{'max_iter':260,'max_leaf_nodes':15,'min_samples_leaf':30,'l2_regularization':20,'learning_rate':.05}},
        'LightGBM':{'family':'lgbm','params':{'n_estimators':700,'num_leaves':15,'min_child_samples':60,'learning_rate':.03,'reg_lambda':20}},
        'CatBoost':{'family':'catboost','params':{'iterations':500,'depth':6,'learning_rate':.05,'l2_leaf_reg':20}},
    }


def fit_spec(spec,X,y):
    start=time.perf_counter();X=X.copy();y=np.asarray(y)
    original_n=len(y)
    if spec.get('training_cap') and len(y)>spec['training_cap']:
        idx=np.sort(np.random.default_rng(SEED).choice(len(y),spec['training_cap'],replace=False))
        X=X.iloc[idx].copy();y=y[idx]
    family,params=spec['family'],dict(spec['params'])
    cats=[c for c in CAT if c in X]
    if family=='lgbm':
        vocabulary={c:sorted(X[c].astype(str).unique()) for c in cats}
        for c in cats:X[c]=pd.Categorical(X[c],categories=vocabulary[c])
        model=LGBMRegressor(**params,n_jobs=2,random_state=SEED,verbosity=-1)
        with threadpool_limits(2):model.fit(X,y,categorical_feature=cats)
    elif family=='catboost':
        vocabulary={}
        for c in cats:X[c]=X[c].astype(str)
        model=CatBoostRegressor(**params,loss_function='RMSE',random_seed=SEED,
            thread_count=2,verbose=False,allow_writing_files=False)
        with threadpool_limits(2):model.fit(X,y,cat_features=cats)
    else:
        vocabulary={}
        numeric=[c for c in X if c not in cats]
        pre=ColumnTransformer([
            ('num',Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
                ('scale',StandardScaler()),('float32',FunctionTransformer(np.asarray,kw_args={'dtype':np.float32},feature_names_out='one-to-one'))]),numeric),
            ('cat',OneHotEncoder(handle_unknown='infrequent_if_exist',max_categories=64,
                min_frequency=10,sparse_output=False,dtype=np.float32),cats)])
        constructors={'linear':LinearRegression,'ridge':Ridge,'knn':KNeighborsRegressor,
            'rf':RandomForestRegressor,'hgb':HistGradientBoostingRegressor,'dummy':DummyRegressor}
        if family=='rf':params.update(n_jobs=2,random_state=SEED,max_samples=min(30000,len(y)))
        if family=='hgb':params.update(early_stopping=False,random_state=SEED)
        if family=='knn':params.update(n_jobs=2)
        if family=='dummy':params.update(strategy='mean')
        model=Pipeline([('pre',pre),('model',constructors[family](**params))])
        with threadpool_limits(2):model.fit(X,y)
    return {'family':family,'model':model,'categories':vocabulary,'features':list(X.columns),
        'spec':spec,'fit_pool_rows':original_n,'actual_fit_rows':len(y),
        'seconds':time.perf_counter()-start}


def predict_fit(fit,X):
    if fit['family']=='blend':
        return sum(w*predict_fit(part,X) for w,part in zip(fit['weights'],fit['parts']))
    X=X[fit['features']].copy()
    if fit['family']=='lgbm':
        for c,v in fit['categories'].items():X[c]=pd.Categorical(X[c],categories=v)
    elif fit['family']=='catboost':
        for c in CAT:
            if c in X:X[c]=X[c].astype(str)
    with threadpool_limits(2):p=fit['model'].predict(X)
    # Physical output constraint, fixed before CV. Labels are never capped.
    return np.maximum(0,np.asarray(p,dtype=float))


def temporal_folds(d,dev,number=5):
    dates=np.sort(dev.date_group.unique());folds=[];rows=[]
    for i,(a,b) in enumerate(TimeSeriesSplit(n_splits=number).split(dates),1):
        boundary=pd.Timestamp(dates[b[0]])
        prior=d.date_group.isin(dates[a])
        train=np.flatnonzero(prior&(d.label_available_at<boundary)&d.known_user.eq(1))
        valid=np.flatnonzero(d.date_group.isin(dates[b])&d.known_user.eq(1))
        assert d.loc[train,'label_available_at'].max()<d.loc[valid,'StartDatetime'].min()
        folds.append((train,valid))
        rows.append({'fold':i,'train_identified':len(train),'valid_identified':len(valid),
            'train_start':d.loc[train,'date_group'].min(),'train_end':d.loc[train,'date_group'].max(),
            'valid_start':d.loc[valid,'date_group'].min(),'valid_end':d.loc[valid,'date_group'].max(),
            'purged_all_users':int((prior&(d.label_available_at>=boundary)).sum())})
    return folds,pd.DataFrame(rows)


def evaluate(spec,X,d,folds):
    rows=[];predictions=[]
    for i,(ti,vi) in enumerate(folds,1):
        fit=fit_spec(spec,X.iloc[ti],d.Demand.iloc[ti])
        p=predict_fit(fit,X.iloc[vi])
        rows.append({'fold':i,**metrics(d.Demand.iloc[vi],p),
            'fit_rows':fit['actual_fit_rows'],'fit_seconds':fit['seconds']})
        predictions.append(p)
    return rows,predictions


def block_bootstrap(frame,prediction,baseline,repeats=2000,block_days=7):
    f=frame[['date_group','Demand']].copy();f['p']=prediction;f['b']=baseline
    f['y2']=f.Demand**2;f['sse']=(f.Demand-f.p)**2;f['ae']=(f.Demand-f.p).abs();f['base_ae']=(f.Demand-f.b).abs()
    a=f.groupby('date_group').agg(n=('Demand','size'),y=('Demand','sum'),y2=('y2','sum'),
        sse=('sse','sum'),ae=('ae','sum'),base_ae=('base_ae','sum')).to_numpy()
    rng=np.random.default_rng(SEED);values=[];nd=len(a)
    for _ in range(repeats):
        starts=rng.integers(0,nd,size=int(np.ceil(nd/block_days)))
        ix=np.concatenate([(s+np.arange(block_days))%nd for s in starts])[:nd]
        n,y,y2,sse,ae,bae=a[ix].sum(axis=0);tss=y2-y*y/n
        values.append([1-sse/tss,ae/n,(bae-ae)/n])
    v=np.array(values)
    return {'method':'Circular moving-block bootstrap of observed local dates',
        'repeats':repeats,'block_days':block_days,'seed':SEED,
        'R2_95pct':np.quantile(v[:,0],[.025,.975]).tolist(),
        'MAE_95pct_kWh':np.quantile(v[:,1],[.025,.975]).tolist(),
        'MAE_gain_over_training_mean_95pct_kWh':np.quantile(v[:,2],[.025,.975]).tolist(),
        'limits':'Conditional on this test window and selected model; does not remove selection or prior exploratory exposure.'}


def run_study(d,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    models_dir=output/'models';models_dir.mkdir(exist_ok=True)
    dev,test,purged,cutoff=chronological_split(d)
    train_ids=set(dev.source_row);test_ids=set(test.source_row)
    train_mask=d.source_row.isin(train_ids);test_mask=d.source_row.isin(test_ids)
    folds,fold_table=temporal_folds(d,dev)
    fold_table.to_csv(output/'cv_folds.csv',index=False)
    protocol={'target':'Demand (kWh)','prediction_point':'StartDatetime',
        'primary_population':'Identified users: UserID != 0, fixed before final test scoring',
        'clean_context_rows':len(d),'identified_population_rows':int(d.known_user.sum()),
        'development_context_rows':len(dev),'test_context_rows':len(test),
        'personalized_train_rows':int((train_mask&d.known_user.eq(1)).sum()),
        'personalized_test_rows':int((test_mask&d.known_user.eq(1)).sum()),
        'test_start':cutoff,'purged_at_test_boundary':len(purged),
        'CV_folds':5,'selection_metric':'Unweighted mean temporal CV R2',
        'primary_history_protocol':'Online/prequential feature store: prior labels available after session end + 60s; model weights frozen throughout each evaluation window.',
        'frozen_history_diagnostic':'Same fitted model, only development-pool history during test.',
        'prior_test_exposure':'The same Korean final period appeared in an earlier coarse benchmark. These results are exploratory, not a newly blind test.',
        'anonymous_rule':'ID 0 has no personal history; a separate all-user LightGBM supplies fallback predictions.',
        'knn_cap':12000,'random_forest_bootstrap_cap':30000,
        'target_capping':False,'prediction_floor_kWh':0,
        'sources_compared_in_development':['Caltech legacy 16,117 rows','Korean transactions 72,856 raw rows','Jiaxing 441,077 raw transactions; pilot development scores recorded separately'],
        'seed':SEED}
    (output/'protocol_before_test.json').write_text(json.dumps(protocol,indent=2))
    assert audit_feature_causality(d)
    start=time.time();X,cats=build_features(d,d)
    print('Causal feature replay:',X.shape,'seconds',round(time.time()-start,2),flush=True)
    pd.DataFrame({'feature':X.columns,'category':[c in cats for c in X.columns],
        'source':['prior completed sessions only' if c in HISTORY else 'arrival timestamp or registered charger/account metadata' for c in X.columns]}).to_csv(output/'feature_dictionary.csv',index=False)
    specs=base_specs();all_rows=[];preds={};summary=[]
    def add(name,spec):
        rows,p=evaluate(spec,X,d,folds);preds[name]=p
        all_rows.extend(dict(model=name,**r) for r in rows)
        summary.append({'model':name,'cv_R2':float(np.mean([r['R2'] for r in rows])),
            'cv_R2_sd':float(np.std([r['R2'] for r in rows],ddof=1)),
            'cv_MAE':float(np.mean([r['MAE'] for r in rows])),
            'cv_RMSE':float(np.mean([r['RMSE'] for r in rows]))})
        print('CV',name,summary[-1],flush=True)
    for name,spec in list(specs.items()):add(name,spec)
    spaces={
        'Ridge':{'alpha':[.1,1,10,100,1000,10000]},
        'Random Forest':{'n_estimators':[100,180],'max_depth':[10,16,None],
            'min_samples_leaf':[5,15,40],'max_features':[.65,1.]},
        'HGB':{'max_iter':[180,300,450],'max_leaf_nodes':[7,15,31],
            'min_samples_leaf':[20,60,120],'l2_regularization':[0,20,50],'learning_rate':[.03,.07]},
        'LightGBM':{'n_estimators':[450,750,1100],'num_leaves':[7,15,31,63],
            'min_child_samples':[25,60,120],'learning_rate':[.025,.04,.07],'reg_lambda':[5,20,50]},
        'CatBoost':{'iterations':[400,700],'depth':[5,6,7],'learning_rate':[.035,.06],'l2_leaf_reg':[10,30]},
    }
    shortlist=pd.DataFrame(summary).sort_values('cv_R2',ascending=False)
    shortlist=shortlist[shortlist.model.isin(spaces)].head(2).model.tolist()
    tuning=[]
    for name in shortlist:
        trials=list(ParameterSampler(spaces[name],n_iter=6,random_state=SEED))
        best_score=-np.inf;best_name=None
        for i,params in enumerate(trials,1):
            label=f'{name} search {i}'
            spec={'family':specs[name]['family'],'params':params}
            add(label,spec)
            tuning.append({'family':name,'trial':i,'cv_R2':summary[-1]['cv_R2'],'parameters':json.dumps(params)})
            specs[label]=spec
            if summary[-1]['cv_R2']>best_score:best_score=summary[-1]['cv_R2'];best_name=label
        # Only the best configuration per tuned family is retained for final candidate comparison.
        keep=best_name;label=name+' Tuned';specs[label]=specs[keep];preds[label]=preds[keep]
        summary.append(dict(next(r for r in summary if r['model']==keep),model=label))
    comparison=pd.DataFrame(summary)
    comparison=comparison[~comparison.model.str.contains(' search ')].sort_values(['cv_R2','model'],ascending=[False,True]).reset_index(drop=True)
    pd.DataFrame(all_rows).to_csv(models_dir/'cv_folds_results.csv',index=False)
    pd.DataFrame(tuning).to_csv(models_dir/'tuning_trials.csv',index=False)
    eligible=comparison.model.tolist()
    # CV-only two-model blend; the same folds evaluate the small, declared weight grid.
    pair=[name for name in eligible if name!='Dummy Mean'][:2]
    blend_best=None
    for w in [.25,.5,.75]:
        ps=[w*a+(1-w)*b for a,b in zip(preds[pair[0]],preds[pair[1]])]
        rows=[metrics(d.Demand.iloc[vi],p) for (_,vi),p in zip(folds,ps)]
        r={'model':'CV Blend','cv_R2':float(np.mean([x['R2'] for x in rows])),
            'cv_R2_sd':float(np.std([x['R2'] for x in rows],ddof=1)),
            'cv_MAE':float(np.mean([x['MAE'] for x in rows])),
            'cv_RMSE':float(np.mean([x['RMSE'] for x in rows]))}
        if blend_best is None or r['cv_R2']>blend_best[0]['cv_R2']:blend_best=(r,w,ps)
    comparison=pd.concat([comparison,pd.DataFrame([blend_best[0]])]).sort_values(['cv_R2','model'],ascending=[False,True]).reset_index(drop=True)
    comparison.to_csv(models_dir/'comparison_cv.csv',index=False)
    selected=comparison.iloc[0].model
    selection={'model':selected,'cv_R2':float(comparison.iloc[0].cv_R2),
        'cv_MAE':float(comparison.iloc[0].cv_MAE),'features':list(X.columns),
        'blend_parts':pair,'blend_weights':[blend_best[1],1-blend_best[1]],
        'eligible_specs':{k:specs[k] for k in eligible},'test_errors_used_for_selection':False,
        'protocol_note':protocol['prior_test_exposure']}
    (models_dir/'selection_before_test.json').write_text(json.dumps(selection,indent=2,default=int))
    print('FIXED SELECTION',selected,'CV R2',comparison.iloc[0].cv_R2,flush=True)
    ti=np.flatnonzero(train_mask&d.known_user.eq(1));vi=np.flatnonzero(test_mask&d.known_user.eq(1))
    fit_cache={};test_rows=[];test_predictions={}
    for name in eligible:
        fit=fit_spec(specs[name],X.iloc[ti],d.Demand.iloc[ti]);p=predict_fit(fit,X.iloc[vi])
        test_rows.append({'model':name,'selected_by_cv':name==selected,
            'train_rows':fit['actual_fit_rows'],'test_rows':len(vi),**metrics(d.Demand.iloc[vi],p)})
        test_predictions[name]=p
        if name in pair or name==selected:fit_cache[name]=fit
    blend_fit={'family':'blend','parts':[fit_cache[n] for n in pair],
        'weights':[blend_best[1],1-blend_best[1]],'features':list(X.columns)}
    blend_prediction=predict_fit(blend_fit,X.iloc[vi]);test_predictions['CV Blend']=blend_prediction
    test_rows.append({'model':'CV Blend','selected_by_cv':selected=='CV Blend',
        'train_rows':len(ti),'test_rows':len(vi),**metrics(d.Demand.iloc[vi],blend_prediction)})
    best=blend_fit if selected=='CV Blend' else fit_cache[selected]
    primary_prediction=test_predictions[selected]
    pd.DataFrame(test_rows).to_csv(models_dir/'personalized_test_results.csv',index=False)
    pf=d.iloc[vi][['source_row','StartDatetime','date_group','ChargerID','UserID','Demand']].copy()
    for name,p in test_predictions.items():pf[name]=p
    pf.to_csv(models_dir/'personalized_test_predictions.csv',index=False)
    # Same model, independent frozen-history diagnostic. It does not change selection.
    frozen,_=build_features(test,dev)
    frozen_known=test.known_user.eq(1).to_numpy()
    frozen_p=predict_fit(best,frozen.loc[frozen_known])
    protocol_rows=[{'cohort':'identified users','history':'online/prequential',**metrics(d.Demand.iloc[vi],primary_prediction)},
        {'cohort':'identified users','history':'frozen development history',**metrics(test.loc[frozen_known,'Demand'],frozen_p)}]
    # An all-user service has a declared fallback for anonymous arrivals.
    general=fit_spec(base_specs()['LightGBM'],X.loc[train_mask],d.loc[train_mask,'Demand'])
    all_ix=np.flatnonzero(test_mask);all_p=predict_fit(general,X.iloc[all_ix]);known=d.known_user.iloc[all_ix].eq(1).to_numpy()
    combined=all_p.copy();combined[known]=primary_prediction
    protocol_rows.extend([
        {'cohort':'all users, general LightGBM','history':'online/prequential',**metrics(d.Demand.iloc[all_ix],all_p)},
        {'cohort':'all users, personalized + anonymous fallback','history':'online/prequential',**metrics(d.Demand.iloc[all_ix],combined)},
        {'cohort':'anonymous users only, fallback','history':'online/prequential',**metrics(d.Demand.iloc[all_ix[~known]],all_p[~known])}])
    protocols=pd.DataFrame(protocol_rows);protocols.to_csv(models_dir/'protocol_and_cohort_results.csv',index=False)
    ap=d.iloc[all_ix][['source_row','date_group','UserID','ChargerID','Demand']].copy()
    ap['general_prediction']=all_p;ap['combined_prediction']=combined
    ap.to_csv(models_dir/'all_user_predictions.csv',index=False)
    uncertainty=block_bootstrap(d.iloc[vi],primary_prediction,test_predictions['Dummy Mean'])
    (models_dir/'block_bootstrap.json').write_text(json.dumps(uncertainty,indent=2))
    bundle={'personalized':best,'anonymous_fallback':general,'history_pool_development':dev,
        'selected_model':selected,'protocol':protocol,'features':list(X.columns),
        'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__}}
    joblib.dump(bundle,models_dir/'arrival_model_bundle.joblib',compress=3)
    loaded=joblib.load(models_dir/'arrival_model_bundle.joblib')
    assert np.allclose(predict_fit(loaded['personalized'],X.iloc[vi[:200]]),primary_prediction[:200])
    # Representation ablation, fixed LightGBM, same chronological folds and online availability.
    ablation=[]
    for label,columns in [('Charger and calendar',[c for c in CALENDAR+CAT if c!='UserID']),
                          ('Identity and calendar',CALENDAR+CAT)]:
        rows,_=evaluate(base_specs()['LightGBM'],X[columns],d,folds)
        ablation.append({'feature_set':label,'cv_R2':np.mean([r['R2'] for r in rows]),'cv_MAE':np.mean([r['MAE'] for r in rows])})
    hist_row=comparison[comparison.model=='LightGBM'].iloc[0]
    ablation.append({'feature_set':'Identity, calendar and causal history','cv_R2':hist_row.cv_R2,'cv_MAE':hist_row.cv_MAE})
    pd.DataFrame(ablation).to_csv(models_dir/'feature_ablation.csv',index=False)
    result={'selected_model':selected,'primary_population':'identified users only',
        'test_metrics':metrics(d.Demand.iloc[vi],primary_prediction),'cv_R2':selection['cv_R2'],
        'raw_rows':72856,'clean_context_rows':len(d),'identified_rows':int(d.known_user.sum()),
        'train_identified':len(ti),'test_identified':len(vi),'test_context':len(test),
        'features':len(X.columns),'test_start':cutoff,'bootstrap':uncertainty,
        'results_exploratory':True}
    (output/'manifest.json').write_text(json.dumps(result,indent=2))
    print('FINAL',json.dumps(result,indent=2),flush=True)
    print(protocols.to_string(index=False),flush=True)
    return result,comparison,pd.DataFrame(test_rows),protocols
