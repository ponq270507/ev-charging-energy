"""Verify added review diagnostics without refitting estimators."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from feedback_checks import decision_checks,ROOT,OUT

def verify():
    protocol=json.loads((OUT/'matched_protocol.json').read_text())
    for key,path in [('source_sha256',ROOT/'data/ChargingRecords_original.csv'),('code_sha256',ROOT/'feedback_checks.py')]:
        assert protocol[key]==hashlib.sha256(path.read_bytes()).hexdigest(),key
    f=pd.read_csv(OUT/'matched_folds.csv');s=pd.read_csv(OUT/'matched_comparison.csv').set_index('model')
    assert len(f)==35 and f.groupby('model').size().eq(5).all()
    for fold,g in f.groupby('fold'):
        assert g.fit_rows.nunique()==g.valid_rows.nunique()==g.encoded_columns.nunique()==1
        assert g.fit_rows.iloc[0]==[3078,6418,11737,17079,23034][fold-1]
        assert g.valid_rows.iloc[0]==[3348,5323,5333,5969,6730][fold-1]
    checks=[]
    for name,g in f.groupby('model'):
        values={'cv_R2':g.R2.mean(),'cv_R2_sd':g.R2.std(ddof=1),'cv_MAE':g.MAE.mean(),'cv_RMSE':g.RMSE.mean()}
        for k,v in values.items():
            assert np.isclose(v,s.loc[name,k],atol=1e-12)
            checks.append({'item':name+' '+k,'computed':v,'rounded_report':round(v,3),'matches':True})
    original=pd.read_csv(OUT/'decision_and_tail.csv')
    recomputed=pd.DataFrame(decision_checks())
    pd.testing.assert_frame_equal(original,recomputed,check_dtype=False,rtol=1e-12)
    for row in recomputed.itertuples():
        assert row.high_energy_n+row.low_energy_n==row.n
        expected=(row.high_energy_n*row.high_energy_MAE+row.low_energy_n*row.low_energy_MAE)/row.n
        source=ROOT/'analysis_outputs/models/personalized_test_results.csv' if row.model=='CV Blend' else ROOT/'analysis_outputs/revision/user_mean_test.csv'
        known=pd.read_csv(source).set_index('model').loc[row.model,'MAE']
        assert np.isclose(expected,known,atol=1e-12)
    result={'matched_fold_fits':35,'common_rows_and_encoding_verified':True,
            'matched_summary_values_verified':len(checks),'ranking_and_tail_recomputed':True,
            'tail_weighted_MAE_reconciled':True,'primary_selection_unchanged':True}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2))
    pd.DataFrame(checks).to_csv(OUT/'matched_summary_checks.csv',index=False)
    return result

if __name__=='__main__':print(json.dumps(verify(),indent=2))
