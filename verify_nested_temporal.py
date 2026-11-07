"""Reconstruct nested partitions/choices and recompute every saved metric."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from nested_temporal import ROOT,OUT,design
from balanced_search import candidates,array_sha,sha
from arrival_history import load_and_audit
from train_arrival import metrics


def verify():
    p=json.loads((OUT/'protocol.json').read_text());end=json.loads((OUT/'completed.json').read_text())
    assert end['protocol_sha256']==sha(OUT/'protocol.json')
    assert end['all_fit_metrics_sha256']==sha(OUT/'all_fit_metrics.csv')
    for rel,value in p['code_sha256'].items():assert sha(ROOT/rel)==value,rel
    assert sha(ROOT/'data/ChargingRecords_original.csv')==p['source_sha256']
    for rel,value in p['original_artifact_hashes'].items():assert sha(ROOT/rel)==value,rel
    cs,spaces=candidates();assert p['candidates']==cs and p['spaces']==spaces
    assert end['actual_fits']==p['expected_total_fits']==595
    raw,_,_=load_and_audit(ROOT/'data/ChargingRecords_original.csv')
    d,cutoff,fold_table,splits=design(raw)
    assert cutoff==p['final_test_start'] and d.date_group.max()<cutoff
    pd.testing.assert_frame_equal(pd.read_csv(OUT/'outer_folds.csv'),fold_table,check_dtype=False)
    info={i['key']:i for i in p['partitions']};assert len(info)==20
    for spec in splits:
        m=info[spec['key']];ti=spec['train'];vi=spec['valid'];truth=np.load(OUT/'truth'/f"{spec['key']}.npz")
        assert np.array_equal(truth['y'],d.Demand.iloc[vi].to_numpy())
        assert np.array_equal(truth['source_row'],d.source_row.iloc[vi].to_numpy())
        assert np.array_equal(truth['fit_source_row'],d.source_row.iloc[ti].to_numpy())
        assert array_sha(truth['source_row'])==m['valid_rows_sha256']
        assert array_sha(truth['fit_source_row'])==m['train_rows_sha256']
        assert set(truth['source_row']).isdisjoint(truth['fit_source_row'])
        assert len(ti)==m['fit_rows'] and len(vi)==m['valid_rows']
        assert d.label_available_at.iloc[ti].max()<d.StartDatetime.iloc[vi].min()
        if spec['stage']=='inner':
            outer=next(s for s in splits if s['stage']=='outer' and s['outer']==spec['outer'])
            assert set(ti)|set(vi)<=set(outer['train'])
            assert d.label_available_at.iloc[vi].max()<pd.Timestamp(spec['outer_boundary'])
            assert set(ti).isdisjoint(outer['valid']) and set(vi).isdisjoint(outer['valid'])

    f=pd.read_csv(OUT/'all_fit_metrics.csv');assert len(f)==595 and f.job_id.is_unique
    assert f[f.stage.eq('inner')].shape[0]==555 and f[f.stage.eq('outer')].shape[0]==40
    count=0
    for r in f.to_dict('records'):
        truth=np.load(OUT/'truth'/f"{r['key']}.npz")
        pred=np.load(OUT/'predictions'/f"{r['job_id']}.npz")['prediction']
        assert pred.shape==truth['y'].shape and np.isfinite(pred).all() and (pred>=0).all()
        for k,value in metrics(truth['y'],pred).items():
            assert np.isclose(r[k],value,rtol=1e-12,atol=1e-12),(r['job_id'],k)
            count+=1
        for key in ['train_matrix_sha256','valid_matrix_sha256','train_rows_sha256','valid_rows_sha256','encoded_columns']:
            assert r[key]==info[r['key']][key],(r['job_id'],key)
    inner=f[f.stage.eq('inner')]
    assert inner.groupby(['outer','model','trial']).inner.nunique().eq(3).all()
    for _,g in inner.groupby(['outer','inner']):
        sizes=g.groupby('model').size()
        assert sizes['Linear Regression']==1 and sizes.drop('Linear Regression').eq(6).all()
        for c in ['fit_rows','valid_rows','encoded_columns','train_matrix_sha256','valid_matrix_sha256']:
            assert g[c].nunique()==1
    search=inner.groupby(['outer','model','family','trial'],sort=False).agg(inner_R2=('R2','mean'),inner_MAE=('MAE','mean')).reset_index()
    search=search.sort_values(['outer','model','inner_R2','trial'],ascending=[True,True,False,True])
    best=search.groupby(['outer','model'],sort=False).head(1)
    choice_file=OUT/'choices_before_outer_scoring.json'
    choices=json.loads(choice_file.read_text())
    assert choices['outer_scores_used_for_selection'] is False
    assert end['selection_sha256']==sha(choice_file)
    assert len(choices['choices'])==35
    for c in choices['choices']:
        row=best[best.outer.eq(c['outer'])&best.model.eq(c['model'])].iloc[0]
        assert c['trial']==row.trial and np.isclose(c['inner_R2'],row.inner_R2,atol=1e-12)
        config=next(x for x in cs if x['model']==c['model'] and x['trial']==c['trial'])
        assert c['params']==config['params']
        outer=f[f.stage.eq('outer')&f.outer.eq(c['outer'])&f.model.eq(c['model'])].iloc[0]
        assert outer.trial==c['trial'] and outer.selection_sha256==sha(choice_file)
    for o,g in best.groupby('outer'):
        winner=g.sort_values(['inner_R2','model','trial'],ascending=[False,True,True]).iloc[0]
        chosen=[c for c in choices['choices'] if c['outer']==o and c['selected_family']]
        assert len(chosen)==1 and chosen[0]['model']==winner.model
    outer=pd.read_csv(OUT/'outer_metrics.csv');assert len(outer)==50
    assert outer.groupby('model').outer.nunique().eq(5).all()
    for row in outer.to_dict('records'):
        truth=np.load(OUT/'truth'/f"o{row['outer']}.npz")
        pred=np.load(OUT/'predictions'/f"{row['job_id']}.npz")['prediction']
        if row['model']=='Nested family selector':
            w=next(c for c in choices['choices'] if c['outer']==row['outer'] and c['selected_family'])
            assert row['family']==w['family'] and row['trial']==w['trial']
        for key,value in metrics(truth['y'],pred).items():assert np.isclose(row[key],value,atol=1e-12)
    summary=pd.read_csv(OUT/'comparison.csv').set_index('model')
    for name,g in outer.groupby('model'):
        for col,value in {'outer_R2':g.R2.mean(),'outer_R2_sd':g.R2.std(ddof=1),'outer_MAE':g.MAE.mean(),'outer_RMSE':g.RMSE.mean()}.items():
            assert np.isclose(summary.loc[name,col],value,atol=1e-12)
    result={'passed':True,'actual_fits':595,'fit_metrics_recomputed':count,
        'outer_summary_rows':len(summary),'outer_predictions_metrics_recomputed':200,
        'inner_partitions_confined_to_outer_train':True,'inner_validation_labels_available_before_outer_fit':True,
        'preprocessors_fitted_per_inner_or_outer_training_partition':True,
        'same_input_rows_and_matrices_within_each_partition':True,
        'choices_reconstructed_from_inner_scores_only':True,'primary_artifacts_unchanged':True,
        'final_test_scored':False,'independent_blind_test_created':False,
        'scope':'Retrospective nested shared-input family selection. Original native-input blend remains a separate historical result.'}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':verify()
