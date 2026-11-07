"""Tables and plots corresponding to the revised PDF; no model selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from arrival_history import CALENDAR, CAT, HISTORY

BLUE, RED = '#2F6690', '#B23A48'


def report_tables(root='.'):
    root=Path(root);out=root/'analysis_outputs';rev=out/'revision';models=out/'models'
    cv=pd.concat([pd.read_csv(models/'comparison_cv.csv'),pd.read_csv(rev/'user_mean_cv.csv')],ignore_index=True)
    cv=cv.sort_values('cv_R2',ascending=False,kind='stable').reset_index(drop=True)
    test=pd.concat([pd.read_csv(models/'personalized_test_results.csv'),pd.read_csv(rev/'user_mean_test.csv')],ignore_index=True)
    test=test.set_index('model').loc[cv.model].reset_index()
    cv.to_csv(rev/'report_table3_cv.csv',index=False)
    test.to_csv(rev/'report_table4_test.csv',index=False)
    return cv,test


def save(fig,folder,name):
    from io import BytesIO
    from PIL import Image
    folder.mkdir(parents=True,exist_ok=True)
    fig.tight_layout()
    for extension in ['png','pdf']:
        buffer=BytesIO()
        fig.savefig(buffer,format=extension,dpi=160,bbox_inches='tight')
        payload=buffer.getvalue()
        if extension=='png':
            with Image.open(BytesIO(payload)) as im:im.verify()
        else:
            assert payload.startswith(b'%PDF') and b'%%EOF' in payload[-2048:]
        target=folder/(name+'.'+extension)
        temporary=target.with_suffix(target.suffix+'.tmp')
        temporary.write_bytes(payload)
        temporary.replace(target)
    plt.show()
    plt.close(fig)


def report_figures(root='.'):
    root=Path(root);out=root/'analysis_outputs';rev=out/'revision';folder=out/'figures'
    h=pd.read_csv(rev/'hour_mean_intervals.csv')
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    ax[0].bar(h.arrival_hour,h.sessions,color=BLUE)
    ax[0].set(title='(a) Arrival frequency',xlabel='Recorded Korean arrival hour',ylabel='Sessions',ylim=(0,7200))
    ax[1].plot(h.arrival_hour,h.mean_kWh,'o-',color=BLUE,ms=4)
    ax[1].fill_between(h.arrival_hour,h.lower_95_kWh,h.upper_95_kWh,color=BLUE,alpha=.18,label='Pointwise 95% block interval')
    low=h.sessions.lt(500)
    ax[1].scatter(h.loc[low,'arrival_hour'],h.loc[low,'mean_kWh'],marker='^',facecolors='white',edgecolors=BLUE,zorder=4)
    ax[1].set(title='(b) Eventual session energy',xlabel='Recorded Korean arrival hour',ylabel='Mean energy (kWh)',ylim=(0,35))
    ax[1].legend(fontsize=8)
    for a in ax:a.set_xticks([0,6,12,18,23])
    save(fig,folder,'report_figure2_hour_intervals')
    selected=json.loads((out/'manifest.json').read_text())['selected_model']
    p=pd.read_csv(out/'models/personalized_test_predictions.csv')
    y=p.Demand.to_numpy();pred=p[selected].to_numpy();res=y-pred
    fig,ax=plt.subplots(1,2,figsize=(12,4.8))
    a=ax[0].hexbin(y,pred,gridsize=55,mincnt=1,norm=LogNorm(),cmap='viridis',extent=(0,100,0,100))
    ax[0].plot([0,100],[0,100],'--',color=RED,lw=1)
    ax[0].set(title='(a) Prediction density',xlabel='Actual energy (kWh)',ylabel='Predicted energy (kWh)',xlim=(0,100),ylim=(0,100))
    fig.colorbar(a,ax=ax[0],label='Sessions per hexagon (log scale)',orientation='horizontal',pad=.2)
    b=ax[1].hexbin(pred,res,gridsize=55,mincnt=1,norm=LogNorm(),cmap='viridis',extent=(0,100,-100,100))
    ax[1].axhline(0,ls='--',color=RED,lw=1);ax[1].plot([0,100],[0,-100],':',color='gray',lw=1)
    bin_index=np.floor(pred/5).astype(int)
    mid=[];med=[]
    for k in sorted(set(bin_index)):
        mask=bin_index==k
        if mask.sum()>=20:mid.append(5*k+2.5);med.append(np.median(res[mask]))
    ax[1].plot(mid,med,'o-',color=RED,ms=4,label='Median, bins with n ≥ 20')
    ax[1].set(title='(b) Residual pattern',xlabel='Predicted energy (kWh)',ylabel='Actual − predicted (kWh)',xlim=(0,100),ylim=(-100,100))
    ax[1].legend(fontsize=8)
    fig.colorbar(b,ax=ax[1],label='Sessions per hexagon (log scale)',orientation='horizontal',pad=.2)
    save(fig,folder,'report_figure3_hexbin_residuals')
    perm=pd.read_csv(rev/'permutation_10.csv').iloc[::-1]
    labels=perm.feature_group.str.replace('_','–',regex=False).str.capitalize()
    fig,ax=plt.subplots(figsize=(11,5))
    ax.barh(labels,perm.mean_R2_decrease,xerr=perm.shuffle_sd,color=BLUE,capsize=3)
    for y0,(v,sd) in enumerate(zip(perm.mean_R2_decrease,perm.shuffle_sd)):
        label=f'{v:.3f}' if abs(v)>=.001 else '<0.001'
        ax.text(max(0,v+sd)+.008,y0,label,va='center',fontsize=9)
    ax.set(xlabel='Decrease in test R² after group permutation',title='2,500 identified sessions; 10 shuffles per group',xlim=(-.01,.4))
    ax.axvline(0,color='black',lw=.6)
    save(fig,folder,'report_figure4_permutation10')
    ab=pd.read_csv(rev/'feature_ablation_extended.csv').iloc[::-1]
    fig,ax=plt.subplots(figsize=(11,4.8))
    ax.barh(ab.feature_set,ab.cv_R2,color=BLUE)
    ax.set(xlabel='Mean temporal CV R²',title='Table 8: fixed-LightGBM representation ablation')
    save(fig,folder,'report_table5_ablation')
    delay=pd.read_csv(rev/'delay_sensitivity.csv')
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    names=['0 s','60 s','5 min','1 hour','1 day']
    ax[0].plot(names,delay.R2,'o-',color=BLUE);ax[0].set(ylabel='Identified test R²',title='Fixed blend configurations; refit for each delay')
    ax[1].plot(names,delay.MAE,'o-',color=RED);ax[1].set(ylabel='MAE (kWh)',title='Same 10,999 test arrivals')
    for a0 in ax:a0.set_xlabel('Assumed reporting delay')
    save(fig,folder,'report_table7_delay')


def primary_report_checks(root='.'):
    root=Path(root);out=root/'analysis_outputs';rev=out/'revision'
    m=json.loads((out/'manifest.json').read_text())
    actual={'raw_rows':m['raw_rows'],'clean_rows':m['clean_context_rows'],
            'identified_rows':m['identified_rows'],'train_identified':m['train_identified'],
            'test_identified':m['test_identified'],'features':m['features'],**m['test_metrics']}
    expected={'raw_rows':72856,'clean_rows':69570,'identified_rows':40790,
              'train_identified':29775,'test_identified':10999,'features':137,
              'R2':.511,'MAE':6.843,'RMSE':9.799,'MedAE':4.531}
    rows=[]
    for k,v in expected.items():
        tol=0 if isinstance(v,int) else .000500001
        rows.append({'item':k,'computed':actual[k],'report':v,'matches':abs(actual[k]-v)<=tol})
    checks=pd.DataFrame(rows);checks.to_csv(rev/'primary_report_consistency.csv',index=False)
    return checks


def verify_revision_outputs(root='.'):
    """Recompute diagnostic scores from saved rows and baseline from raw history."""
    from arrival_history import load_and_audit,chronological_split,build_features
    from revision_checks import user_mean
    from train_arrival import metrics,predict_fit
    import joblib
    root=Path(root);out=root/'analysis_outputs';rev=out/'revision'
    d,_,_=load_and_audit(root/'data/ChargingRecords_original.csv')
    dev,test,_,_=chronological_split(d)
    query=test[test.known_user.eq(1)].reset_index(drop=True)
    xx,_=build_features(query,d)
    primary=pd.read_csv(out/'models/personalized_test_predictions.csv')
    bundle=joblib.load(out/'models/arrival_model_bundle.joblib')
    np.testing.assert_array_equal(primary.source_row,query.source_row)
    np.testing.assert_allclose(predict_fit(bundle['personalized'],xx),
                               primary[bundle['selected_model']],rtol=1e-10,atol=1e-10)
    expected,personal=user_mean(xx)
    saved=pd.read_csv(rev/'user_mean_predictions.csv')
    np.testing.assert_array_equal(saved.source_row,query.source_row)
    np.testing.assert_allclose(saved['User historical mean'],expected,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(saved.used_personal_history,personal)
    baseline=pd.read_csv(rev/'user_mean_test.csv').iloc[0]
    for k,value in metrics(query.Demand,expected).items():
        assert np.isclose(value,baseline[k]),('User mean metric',k)
    delay=pd.read_csv(rev/'delay_sensitivity.csv')
    for row in delay.itertuples():
        p=pd.read_csv(rev/f'delay_{row.delay_seconds}_predictions.csv')
        np.testing.assert_array_equal(p.source_row,query.source_row)
        np.testing.assert_allclose(p.Demand,query.Demand)
        for k,value in metrics(p.Demand,p.prediction).items():
            assert np.isclose(value,getattr(row,k)),('Delay metric',row.delay_seconds,k)
        changed=d.copy()
        changed['label_available_at']=changed.EndDatetime+pd.Timedelta(seconds=row.delay_seconds)
        train,_,_,_=chronological_split(changed)
        assert int(train.known_user.sum())==row.fit_rows
    folds=pd.read_csv(rev/'user_mean_folds.csv')
    cv=pd.read_csv(rev/'user_mean_cv.csv').iloc[0]
    for k,value in {'cv_R2':folds.R2.mean(),'cv_R2_sd':folds.R2.std(ddof=1),
                    'cv_MAE':folds.MAE.mean(),'cv_RMSE':folds.RMSE.mean()}.items():
        assert np.isclose(value,cv[k]),('User mean CV',k)
    ab=pd.read_csv(rev/'feature_ablation_extended.csv').iloc[-2:]
    detail=pd.read_csv(rev/'ablation_folds.csv')
    for row in ab.itertuples():
        values=detail[detail.feature_set.eq(row.feature_set)]
        assert len(values)==5
        assert np.isclose(values.R2.mean(),row.cv_R2)
        assert np.isclose(values.MAE.mean(),row.cv_MAE)
    dictionary=pd.read_csv(rev/'feature_dictionary_complete.csv')
    assert len(dictionary)==137 and dictionary.feature.is_unique
    result={'primary_predictions_reconstructed_from_raw_history':True,
            'baseline_reconstructed_from_raw_history':True,
            'all_five_delay_scores_recomputed':True,'delay_fit_counts_repurg_checked':True,
            'added_cv_summaries_reconciled':True,'full_feature_dictionary_checked':True}
    (rev/'revision_verification.json').write_text(json.dumps(result,indent=2))
    return result
