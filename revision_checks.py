"""Reproduce the diagnostic additions in Group 25's revised report.

These post-selection checks never alter the CV-selected primary model.
Use --force to recompute. The default verifies and reuses a complete cache.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import joblib
import numpy as np
import pandas as pd
from arrival_history import (load_and_audit, chronological_split, build_features,
                             CAT, CALENDAR, HISTORY, STAT_NAMES)
from train_arrival import (base_specs, evaluate, temporal_folds, fit_spec,
                           predict_fit, metrics, block_bootstrap)
from importance_diagnostics import run_importance

SEED = 42
FILES = ['user_mean_cv.csv', 'user_mean_test.csv', 'user_mean_predictions.csv',
         'feature_ablation_extended.csv', 'delay_sensitivity.csv',
         'paired_mae_bootstrap.csv', 'permutation_10.csv',
         'hour_mean_intervals.csv', 'history_statistics.csv',
         'feature_dictionary_complete.csv', 'fold_reconciliation.json',
         'revision_manifest.json']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def user_mean(X):
    """No fit: available account history, otherwise available global history."""
    personal = X.user_log_count.gt(0) & X.user_energy_mean.notna()
    prediction = X.user_energy_mean.where(personal, X.global_energy_mean)
    if prediction.isna().any():
        raise ValueError('No available global history for an evaluation query.')
    return prediction.to_numpy(float), personal.to_numpy()


def rounded_report_checks(rev):
    """Compare computed results to displayed report values, never generate scores."""
    checks = []
    def compare(label, actual, expected, tolerance=.000500001):
        checks.append({'item':label, 'computed':float(actual),
                       'report':expected, 'matches':bool(abs(actual-expected)<=tolerance)})
    cv = pd.read_csv(rev/'user_mean_cv.csv').iloc[0]
    for k,v in {'cv_R2':.451,'cv_R2_sd':.032,'cv_MAE':6.993,'cv_RMSE':10.119}.items():
        compare('User mean '+k,cv[k],v)
    test = pd.read_csv(rev/'user_mean_test.csv').iloc[0]
    for k,v in {'R2':.377,'MAE':7.809,'RMSE':11.066,'MedAE':5.175}.items():
        compare('User mean '+k,test[k],v)
    compare('User history available',test.personal_history_rows,10439,0)
    compare('User history fallback',test.global_fallback_rows,560,0)
    ab = pd.read_csv(rev/'feature_ablation_extended.csv').set_index('feature_set')
    for name,n,r2,mae in [
        ('All history, without raw UserID',136,.534,6.529),
        ('Nonpersonal history, without UserID',76,.300,8.283)]:
        compare(name+' columns',ab.loc[name,'columns'],n,0)
        compare(name+' R2',ab.loc[name,'cv_R2'],r2)
        compare(name+' MAE',ab.loc[name,'cv_MAE'],mae)
    delay = pd.read_csv(rev/'delay_sensitivity.csv').set_index('delay_seconds')
    for seconds,n,r2,mae in [(0,29775,.514,6.831),(60,29775,.511,6.843),
                            (300,29775,.510,6.863),(3600,29772,.507,6.902),
                            (86400,29618,.499,7.007)]:
        compare(f'Delay {seconds} fit rows',delay.loc[seconds,'fit_rows'],n,0)
        compare(f'Delay {seconds} R2',delay.loc[seconds,'R2'],r2)
        compare(f'Delay {seconds} MAE',delay.loc[seconds,'MAE'],mae)
    paired=pd.read_csv(rev/'paired_mae_bootstrap.csv').set_index('baseline')
    for name,gain,lo,hi in [('User historical mean',.966,.878,1.072),
                            ('CatBoost',.011,.001,.020),
                            ('Dummy Mean',4.147,4.026,4.266)]:
        # Dummy point gain is not quoted to three decimals in the report.
        if name!='Dummy Mean':compare(name+' MAE gain',paired.loc[name,'gain_kWh'],gain)
        compare(name+' gain CI lower',paired.loc[name,'lower_95_kWh'],lo)
        compare(name+' gain CI upper',paired.loc[name,'upper_95_kWh'],hi)
    perm=pd.read_csv(rev/'permutation_10.csv').set_index('feature_group')
    for name,value in [('user history',.304),('user_charger history',.085),
                       ('Calendar',.040),('charger history',.022)]:
        compare('Permutation '+name,perm.loc[name,'mean_R2_decrease'],value)
    out=pd.DataFrame(checks);out.to_csv(rev/'report_consistency.csv',index=False)
    return out


def run_revision_checks(root='.', force=False):
    root=Path(root).resolve();out=root/'analysis_outputs';models=out/'models'
    rev=out/'revision';rev.mkdir(parents=True,exist_ok=True)
    source_hash=sha256(root/'data/ChargingRecords_original.csv')
    model_hash=sha256(models/'arrival_model_bundle.joblib')
    code_hash=sha256(__file__)
    if not force and all((rev/n).is_file() for n in FILES):
        cached=json.loads((rev/'revision_manifest.json').read_text())
        if (cached.get('source_sha256')==source_hash and
            cached.get('primary_model_sha256')==model_hash and
            cached.get('revision_code_sha256')==code_hash):
            print('Verified revision cache: source, model and code hashes agree.',flush=True)
            return cached
    start=time.time()
    print('Recomputing report diagnostics; primary selection stays fixed.',flush=True)
    d,_,_=load_and_audit(root/'data/ChargingRecords_original.csv')
    dev,test,purged,cutoff=chronological_split(d)
    folds,fold_table=temporal_folds(d,dev)
    X,_=build_features(d,d)
    test_mask=d.source_row.isin(test.source_row)&d.known_user.eq(1)
    query=d.loc[test_mask];Xtest=X.loc[test_mask]
    p=pd.read_csv(models/'personalized_test_predictions.csv')
    assert np.array_equal(query.source_row.to_numpy(),p.source_row.to_numpy())
    bundle=joblib.load(models/'arrival_model_bundle.joblib')
    selected=bundle['selected_model']
    reconstructed=predict_fit(bundle['personalized'],Xtest)
    np.testing.assert_allclose(reconstructed,p[selected],rtol=1e-10,atol=1e-10)

    # Stronger history-only baseline; it was not eligible for model selection.
    mean_pred,personal=user_mean(Xtest)
    cv_rows=[]
    for i,(_,vi) in enumerate(folds,1):
        pp,_=user_mean(X.iloc[vi])
        cv_rows.append({'fold':i,**metrics(d.Demand.iloc[vi],pp)})
    cv_f=pd.DataFrame(cv_rows);cv_f.to_csv(rev/'user_mean_folds.csv',index=False)
    cv_summary={'model':'User historical mean',
                'cv_R2':cv_f.R2.mean(),'cv_R2_sd':cv_f.R2.std(ddof=1),
                'cv_MAE':cv_f.MAE.mean(),'cv_RMSE':cv_f.RMSE.mean()}
    pd.DataFrame([cv_summary]).to_csv(rev/'user_mean_cv.csv',index=False)
    mean_score={'model':'User historical mean',**metrics(query.Demand,mean_pred),
                'test_rows':len(query),'personal_history_rows':int(personal.sum()),
                'global_fallback_rows':int((~personal).sum())}
    pd.DataFrame([mean_score]).to_csv(rev/'user_mean_test.csv',index=False)
    pred_frame=p[['source_row','date_group','Demand']].copy()
    pred_frame['User historical mean']=mean_pred
    pred_frame['used_personal_history']=personal
    pred_frame.to_csv(rev/'user_mean_predictions.csv',index=False)
    print('User historical mean:',mean_score,flush=True)

    # Exactly the same initial LightGBM and folds as original ablation.
    ab=pd.read_csv(models/'feature_ablation.csv').copy()
    ab['columns']=[16,17,137]
    ab_rows=ab.to_dict('records')
    fold_rows=[]
    personal_columns={prefix+'_'+name for prefix in ['user','user_weekday','user_charger'] for name in STAT_NAMES}
    sets=[('All history, without raw UserID',[c for c in X if c!='UserID']),
          ('Nonpersonal history, without UserID',[c for c in X if c!='UserID' and c not in personal_columns])]
    for label,columns in sets:
        rows,_=evaluate(base_specs()['LightGBM'],X[columns],d,folds)
        ab_rows.append({'feature_set':label,'columns':len(columns),
                        'cv_R2':np.mean([r['R2'] for r in rows]),
                        'cv_MAE':np.mean([r['MAE'] for r in rows])})
        fold_rows.extend(dict(feature_set=label,**r) for r in rows)
        print('Ablation:',ab_rows[-1],flush=True)
    pd.DataFrame(ab_rows).to_csv(rev/'feature_ablation_extended.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(rev/'ablation_folds.csv',index=False)

    # Delay changes rebuild histories and purge fitting labels before refitting.
    selection=json.loads((models/'selection_before_test.json').read_text())
    delay_rows=[]
    for seconds in [0,60,300,3600,86400]:
        changed=d.copy()
        changed['label_available_at']=changed.EndDatetime+pd.Timedelta(seconds=seconds)
        dev_s,test_s,purged_s,cutoff_s=chronological_split(changed)
        assert cutoff_s==cutoff and np.array_equal(test_s.source_row,test.source_row)
        ti=changed.source_row.isin(dev_s.source_row)&changed.known_user.eq(1)
        vi=changed.source_row.isin(test_s.source_row)&changed.known_user.eq(1)
        if seconds==60:
            pp=p[selected].to_numpy()
        else:
            xx,_=build_features(changed,changed)
            parts=[fit_spec(selection['eligible_specs'][name],xx.loc[ti],changed.loc[ti,'Demand'])
                   for name in selection['blend_parts']]
            blend={'family':'blend','parts':parts,'weights':selection['blend_weights'],
                   'features':list(xx.columns)}
            pp=predict_fit(blend,xx.loc[vi])
        saved=changed.loc[vi,['source_row','date_group','Demand']].copy()
        saved['prediction']=pp;saved.to_csv(rev/f'delay_{seconds}_predictions.csv',index=False)
        delay_rows.append({'delay_seconds':seconds,'fit_rows':int(ti.sum()),
                           'test_rows':int(vi.sum()),'purged_all_users':len(purged_s),
                           **metrics(changed.loc[vi,'Demand'],pp)})
        pd.DataFrame(delay_rows).to_csv(rev/'delay_sensitivity.csv',index=False)
        print('Delay:',delay_rows[-1],flush=True)

    paired=[]
    for name,baseline in [('Dummy Mean',p['Dummy Mean']),
                          ('User historical mean',mean_pred),('CatBoost',p['CatBoost'])]:
        ci=block_bootstrap(query,p[selected].to_numpy(),np.asarray(baseline))
        lo,hi=ci['MAE_gain_over_training_mean_95pct_kWh']
        paired.append({'baseline':name,'gain_kWh':float(np.abs(query.Demand.to_numpy()-baseline).mean()-np.abs(query.Demand.to_numpy()-p[selected]).mean()),
                       'lower_95_kWh':lo,'upper_95_kWh':hi,'draws':2000,'block_dates':7})
    pd.DataFrame(paired).to_csv(rev/'paired_mae_bootstrap.csv',index=False)
    perm=run_importance(root,sample_size=2500,repeats=10)
    perm.to_csv(rev/'permutation_10.csv',index=False)
    print('Permutation checks completed (10 shuffles per group).',flush=True)

    # Hourly descriptive intervals from circular seven-observed-date blocks.
    dates=sorted(d.date_group.unique());hours=np.arange(24)
    grouped=d.assign(arrival_hour=d.StartDatetime.dt.hour).groupby(['date_group','arrival_hour']).Demand.agg(['sum','count'])
    idx=pd.MultiIndex.from_product([dates,hours],names=['date_group','arrival_hour'])
    grid=grouped.reindex(idx,fill_value=0)
    sums=grid['sum'].to_numpy().reshape(len(dates),24)
    counts=grid['count'].to_numpy().reshape(len(dates),24)
    rng=np.random.default_rng(SEED);samples=[];nd=len(dates)
    for _ in range(2000):
        starts=rng.integers(0,nd,size=int(np.ceil(nd/7)))
        ix=np.concatenate([(v+np.arange(7))%nd for v in starts])[:nd]
        n=counts[ix].sum(axis=0)
        samples.append(np.divide(sums[ix].sum(axis=0),n,out=np.full(24,np.nan),where=n>0))
    boot=np.asarray(samples);n=counts.sum(axis=0)
    pd.DataFrame({'arrival_hour':hours,'sessions':n,'mean_kWh':sums.sum(axis=0)/n,
                  'lower_95_kWh':np.nanquantile(boot,.025,axis=0),
                  'upper_95_kWh':np.nanquantile(boot,.975,axis=0)}).to_csv(rev/'hour_mean_intervals.csv',index=False)

    meanings=[
        ('log_count','log1p(completed count)','dimensionless'),
        ('energy_mean','cumulative completed energy mean','kWh'),
        ('energy_std','population SD of completed energy','kWh'),
        ('energy_last','most recently available energy','kWh'),
        ('energy_last3_mean','mean of up to last 3 energies','kWh'),
        ('energy_last10_mean','mean of up to last 10 energies','kWh'),
        ('energy_last10_median','median of up to last 10 energies','kWh'),
        ('duration_mean','cumulative mean connection duration','hours'),
        ('duration_last','most recent connection duration','hours'),
        ('duration_last3_mean','mean of up to last 3 connection durations','hours'),
        ('duration_last10_median','median of up to last 10 connection durations','hours'),
        ('power_last10_median','median energy/connection-hours over last 10','kW proxy'),
        ('power_last3_mean','mean energy/connection-hours over last 3','kW proxy'),
        ('historical_max_energy','maximum eligible completed energy','kWh'),
        ('historical_min_energy','minimum eligible completed energy','kWh'),
        ('log_history_span_days','log1p(days from first to last availability)','dimensionless'),
        ('mean_cycle_days','availability span/(count-1), if positive','days'),
        ('days_since_label','arrival minus latest label availability','days'),
        ('relative_cycle_gap','days_since_label/mean_cycle_days','dimensionless'),
        ('gap_energy_proxy','min(max energy, mean energy * relative gap)','kWh heuristic')]
    assert [v[0] for v in meanings]==STAT_NAMES
    pd.DataFrame(meanings,columns=['suffix','definition','unit']).to_csv(rev/'history_statistics.csv',index=False)
    definitions={v[0]:(v[1],v[2]) for v in meanings};dictionary=[]
    for c in X:
        entry={'feature':c,'type':'categorical' if c in CAT else 'numeric',
               'availability':'arrival timestamp or registered metadata','definition':c,'unit':'category' if c in CAT else 'calendar/status'}
        for prefix in ['global','charger','user','charger_weekday','user_weekday','user_charger']:
            for suffix,(meaning,unit) in definitions.items():
                if c==prefix+'_'+suffix:
                    entry.update(availability='previous EndDatetime + reporting delay < arrival',
                                 definition=prefix+': '+meaning,unit=unit)
        dictionary.append(entry)
    pd.DataFrame(dictionary).to_csv(rev/'feature_dictionary_complete.csv',index=False)
    last_boundary=pd.Timestamp(fold_table.iloc[-1].valid_start)
    fold5purged=int(((d.date_group<fold_table.iloc[-1].valid_start)&
                      (d.label_available_at>=last_boundary)&d.known_user.eq(1)).sum())
    reconciliation={'fold5_fit_identified':23034,'fold5_valid_identified':6730,
                    'fold5_purged_identified':fold5purged,
                    'final_purged_identified':int(purged.known_user.sum()),
                    'final_fit_identified':int(dev.known_user.sum())}
    assert 23034+6730+fold5purged-int(purged.known_user.sum())==int(dev.known_user.sum())
    (rev/'fold_reconciliation.json').write_text(json.dumps(reconciliation,indent=2))
    consistency=rounded_report_checks(rev)
    manifest={'source_sha256':source_hash,'primary_model_sha256':model_hash,
              'revision_code_sha256':code_hash,'original_selection':selected,
              'test_rows':len(query),'original_primary_predictions_reconstructed':True,
              'report_checks':len(consistency),'report_checks_matching':int(consistency.matches.sum()),
              'all_report_checks_match':bool(consistency.matches.all()),
              'diagnostics_post_selection':True,'seconds':time.time()-start,
              'note':'Computed outputs, never copied from PDF. Disagreements are saved in report_consistency.csv.'}
    (rev/'revision_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(Path(__file__).resolve().parent))
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    run_revision_checks(args.root,force=args.force)
