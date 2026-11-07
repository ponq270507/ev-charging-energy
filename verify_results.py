from pathlib import Path
import sys,json,ast
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
from arrival_history import load_and_audit,chronological_split,build_features,audit_feature_causality
from train_arrival import predict_fit,metrics
from predict_arrival import predict_sessions

o=root/'analysis_outputs';models=o/'models'
d,q,a=load_and_audit(root/'data/ChargingRecords_original.csv')
dev,test,purged,cutoff=chronological_split(d)
manifest=json.loads((o/'manifest.json').read_text())
selected=manifest['selected_model']
selection=json.loads((models/'selection_before_test.json').read_text())
cv=pd.read_csv(models/'comparison_cv.csv')
results=pd.read_csv(models/'personalized_test_results.csv')
preds=pd.read_csv(models/'personalized_test_predictions.csv')
bundle=joblib.load(models/'arrival_model_bundle.joblib')
assert selected==selection['model']==cv.iloc[0].model
assert selection['test_errors_used_for_selection'] is False
assert len(preds)==manifest['test_identified']==10999
assert len(d)==69570 and len(q)==3286 and int(d.known_user.sum())==40790
assert set(dev.source_row).isdisjoint(test.source_row)
assert dev.label_available_at.max()<test.StartDatetime.min()
assert audit_feature_causality(d)
for row in results.itertuples():
 calculated=metrics(preds.Demand,preds[row.model])
 for k,v in calculated.items():assert np.isclose(v,getattr(row,k))
assert not set(['Duration','EndDatetime','Demand'])&set(bundle['features'])
assert len(bundle['features'])==137

# The public inference interface accepts arrival inputs without outcomes.
sample=test[test.ChargerID.isin(dev.ChargerID)].head(40).copy()
frozen,_=build_features(sample,dev)
expected=np.empty(len(sample));known=sample.known_user.eq(1).to_numpy()
expected[known]=predict_fit(bundle['personalized'],frozen.loc[known])
expected[~known]=predict_fit(bundle['anonymous_fallback'],frozen.loc[~known])
actual=predict_sessions(models/'arrival_model_bundle.joblib',sample[['ChargerID','UserID','StartDatetime']])
assert np.allclose(expected,actual.predicted_kWh)

# An online feed contains only genuinely completed records before the query.
for flag in [0,1]:
 row=test[test.known_user.eq(flag)&test.ChargerID.isin(dev.ChargerID)].tail(1).copy()
 extra=d[(~d.source_row.isin(dev.source_row))&(d.label_available_at<row.StartDatetime.iloc[0])].copy()
 actual=predict_sessions(models/'arrival_model_bundle.joblib',row[['ChargerID','UserID','StartDatetime']],extra)
 xx,_=build_features(row,d)
 fit=bundle['personalized'] if flag else bundle['anonymous_fallback']
 expected=predict_fit(fit,xx)
 assert np.allclose(expected,actual.predicted_kWh)

# ID availability is fixed at arrival, distinct from having seen an account before.
p=preds.copy();p['account_seen_during_training']=p.UserID.astype(str).isin(dev.UserID.astype(str))
diagnostics=[]
for key,g in p.groupby('account_seen_during_training'):
 diagnostics.append({'account_seen_during_training':bool(key),'sessions':len(g),**metrics(g.Demand,g[selected])})
pd.DataFrame(diagnostics).to_csv(models/'error_by_account_seen_in_training.csv',index=False)
report={'passed':True,'rows_reconciled':True,'CV_selection_checked':True,
 'all_saved_model_scores_recomputed':True,'future_label_perturbation_passed':True,
 'inference_without_current_outcomes_checked':True,'online_completed_feed_checked':True,
 'training_and_test_label_boundary_checked':True}
(o/'verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2));print(pd.DataFrame(diagnostics).to_string(index=False))
