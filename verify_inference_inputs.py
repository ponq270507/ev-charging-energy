"""Exercise inference boundaries; no training or score-based model selection."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import joblib
from arrival_history import load_and_audit,chronological_split
from predict_arrival import predict_sessions

ROOT=Path(__file__).resolve().parent

def verify():
    model=ROOT/'analysis_outputs/models/arrival_model_bundle.joblib'
    source=ROOT/'data/ChargingRecords_original.csv'
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [model,source]}
    d,_,_=load_and_audit(source);dev,test,_,_=chronological_split(d)
    row=test[test.known_user.eq(1)&test.ChargerID.isin(dev.ChargerID)].tail(1)
    q=row[['ChargerID','UserID','StartDatetime']].copy().reset_index(drop=True)
    eligible=d[~d.source_row.isin(dev.source_row)&d.label_available_at.lt(q.StartDatetime.iloc[0])]
    feed=eligible.tail(80).drop(columns='source_row').copy()
    def pred(arrivals=q,history=feed):return predict_sessions(model,arrivals,history).predicted_kWh.to_numpy()
    expected=pred();checks=[]
    # Different representations of one instant, including a mixed-aware batch.
    aware=q.copy();aware.StartDatetime=aware.StartDatetime.dt.tz_localize('Asia/Seoul').dt.tz_convert('UTC')
    aware_feed=feed.copy()
    for c in ['StartDatetime','EndDatetime']:
        aware_feed[c]=aware_feed[c].dt.tz_localize('Asia/Seoul').dt.tz_convert('UTC')
    assert np.allclose(pred(aware,aware_feed),expected,rtol=0,atol=1e-10)
    mixed=pd.concat([q,aware],ignore_index=True)
    assert np.allclose(pred(mixed),expected[0],rtol=0,atol=1e-10)
    checks.append('Korean naive / UTC aware / mixed query instants agree, including the completed feed')
    explicit=q.copy();floating=q.copy()
    for c in ['ChargerCompany','ChargerType','Location']:
        explicit[c]=row[c].iloc[0]
        try:floating[c]=float(row[c].iloc[0])
        except ValueError:floating[c]=row[c].iloc[0]  # Location may be a text label, e.g. resort.
    assert np.allclose(pred(explicit),pred(floating),rtol=0,atol=1e-10)
    checks.append('Equivalent string/integer-valued float metadata codes agree')
    # Reordering/retransmission of cumulative events must not multiply history.
    duplicated=pd.concat([feed,feed,dev.tail(3)],ignore_index=True).sample(frac=1,random_state=42)
    assert np.allclose(pred(history=duplicated),expected,rtol=0,atol=1e-10)
    checks.append('Duplicate/reordered feed and repeated development records are idempotent')
    # A future-starting event is not a source of current metadata or outcomes.
    future=feed.tail(1).copy()
    future['ChargerID']=q.ChargerID.iloc[0];future['UserID']=q.UserID.iloc[0]
    future['StartDatetime']=q.StartDatetime.iloc[0]+pd.Timedelta(days=2)
    future['EndDatetime']=future.StartDatetime+pd.Timedelta(minutes=60)
    future['Duration']=60.;future['Demand']=90.
    for c in ['ChargerCompany','ChargerType','Location']:future[c]='future_unavailable_category'
    assert np.allclose(pred(history=pd.concat([feed,future],ignore_index=True)),expected,rtol=0,atol=1e-10)
    checks.append('Future event labels and metadata cannot change earlier predictions')
    invalid=[]
    for field,value in [('Demand',0),('Demand',np.inf),('Duration',-1),
                         ('StartDatetime',pd.NaT),('EndDatetime',feed.StartDatetime.iloc[0]),
                         ('Duration',feed.Duration.iloc[0]+10)]:
        bad=feed.head(1).copy();bad.loc[bad.index[0],field]=value
        invalid.append((f'completed {field}={value}',q,bad))
    conflicting=feed.head(1).copy();conflicting.Demand+=1
    invalid.append(('conflicting duplicate outcome',q,pd.concat([feed,conflicting])))
    conflicting=feed.head(2).copy();conflicting['source_row']=int(dev.source_row.iloc[0])
    invalid.append(('source_row reused for another event',q,conflicting))
    badq=q.copy();badq.StartDatetime=pd.NaT;invalid.append(('missing arrival time',badq,feed))
    badq=q.copy();badq.StartDatetime=pd.Timestamp('2022-01-01');invalid.append(('arrival before fitting cutoff',badq,feed))
    for label,arrivals,history in invalid:
        try:pred(arrivals,history)
        except ValueError:pass
        else:raise AssertionError('Accepted invalid input: '+label)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==before[p.name] for p in [model,source])
    result={'passed':True,'invariance_checks':checks,'invalid_cases_rejected':[v[0] for v in invalid],
        'source_and_model_unchanged':True,'scope':'Functional inference boundaries; no independent evaluation or retraining'}
    (ROOT/'analysis_outputs/inference_input_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':verify()
