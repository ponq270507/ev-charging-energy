"""Check equivalent identifier representations at the public inference boundary.

These cases previously sent anonymous float 0.0 to the personalized model.
No models are fitted or selected; the raw dataset and saved bundle stay fixed.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from predict_arrival import predict_sessions

ROOT=Path(__file__).resolve().parent


def verify():
    model=ROOT/'analysis_outputs/models/arrival_model_bundle.joblib'
    before=hashlib.sha256(model.read_bytes()).hexdigest()
    data=pd.read_csv(ROOT/'data/ChargingRecords_original.csv',dtype={'UserID':str,'ChargerID':str})
    test=data[data.StartDatetime.ge('2022-07-19')]
    checked=[]
    for anonymous in [True,False]:
        row=test[test.UserID.eq('0') if anonymous else test.UserID.ne('0')].iloc[0]
        user_values=[0,0.,'0','0.0',None,np.nan,''] if anonymous else [row.UserID,int(row.UserID),float(row.UserID)]
        records=[]
        for user in user_values:
            for charger in [row.ChargerID,int(row.ChargerID),float(row.ChargerID)]:
                records.append({'ChargerID':charger,'UserID':user,'StartDatetime':row.StartDatetime})
        # Object dtype preserves each input case instead of pandas coalescing them.
        q=pd.DataFrame(records,dtype=object)
        out=predict_sessions(model,q)
        route='anonymous fallback' if anonymous else 'personalized'
        assert out.model_route.eq(route).all()
        assert out.UserID.eq('0' if anonymous else row.UserID).all()
        assert out.ChargerID.eq(row.ChargerID).all()
        assert np.allclose(out.predicted_kWh,out.predicted_kWh.iloc[0],atol=1e-12)
        checked.append({'anonymous':anonymous,'equivalent_inputs':len(out),'route':route,
            'equal_predictions':True})
    rejected=0
    for change in [{'ChargerID':None},{'UserID':1.5},{'UserID':float('inf')},{'ChargerID':True}]:
        q={'ChargerID':row.ChargerID,'UserID':row.UserID,'StartDatetime':row.StartDatetime,**change}
        try:
            predict_sessions(model,pd.DataFrame([q],dtype=object))
        except ValueError:
            rejected+=1
        else:
            raise AssertionError(f'Invalid ID accepted: {change}')
    assert hashlib.sha256(model.read_bytes()).hexdigest()==before
    result={'passed':True,'equivalent_id_cases':checked,'invalid_inputs_rejected':rejected,
        'model_bundle_unchanged':True,'scope':'ID normalization and routing; no model refit or independent performance evaluation'}
    out=ROOT/'analysis_outputs/inference_id_verification.json'
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':verify()
