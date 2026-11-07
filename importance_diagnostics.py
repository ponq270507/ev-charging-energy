"""Post-selection group permutation sensitivity; no retuning follows it."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from arrival_history import load_and_audit,build_features,CALENDAR,HISTORY
from train_arrival import predict_fit
from study_plots import save_figure,BLUE

def run_importance(root='.',sample_size=2500,repeats=3):
    root=Path(root);out=root/'analysis_outputs';m=out/'models'
    d,_,_=load_and_audit(root/'data/ChargingRecords_original.csv')
    bundle=joblib.load(m/'arrival_model_bundle.joblib')
    saved=pd.read_csv(m/'personalized_test_predictions.csv').sample(min(sample_size,bundle['protocol']['personalized_test_rows']),random_state=42)
    query=d.set_index('source_row').loc[saved.source_row].reset_index()
    X,_=build_features(query,d)
    fit=bundle['personalized'];pred=predict_fit(fit,X);selected=bundle['selected_model']
    assert np.allclose(pred,saved[selected])
    baseline=r2_score(query.Demand,pred)
    groups={'Calendar':CALENDAR,'Charger metadata':['ChargerID','ChargerCompany','Location','ChargerType'],'User identity':['UserID']}
    for prefix in ['global','charger','user','charger_weekday','user_weekday','user_charger']:
        # Exact prefix split avoids collecting user_charger into the generic user group.
        from arrival_history import STAT_NAMES
        groups[prefix+' history']=[prefix+'_'+s for s in STAT_NAMES]
    rng=np.random.default_rng(42);rows=[]
    for label,columns in groups.items():
        changes=[]
        for _ in range(repeats):
            xp=X.copy();ix=rng.permutation(len(X))
            xp[columns]=X[columns].iloc[ix].to_numpy()
            changes.append(baseline-r2_score(query.Demand,predict_fit(fit,xp)))
        rows.append({'feature_group':label,'mean_R2_decrease':np.mean(changes),
            'shuffle_sd':np.std(changes),'sample_rows':len(query),'repeats':repeats})
    table=pd.DataFrame(rows).sort_values('mean_R2_decrease',ascending=False)
    table.to_csv(m/'group_permutation_importance.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,5));v=table.iloc[::-1]
    ax.barh(v.feature_group,v.mean_R2_decrease,xerr=v.shuffle_sd,color=BLUE,capsize=3)
    ax.axvline(0,color='black',linewidth=.7)
    ax.set(title='Grouped input permutation sensitivity',xlabel='Decrease in test R² after group permutation')
    save_figure(fig,out/'figures','17_importance')
    return table

if __name__=='__main__':print(run_importance())
