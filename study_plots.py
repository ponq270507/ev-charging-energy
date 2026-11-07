"""Plot and SQL presentation helpers; original palette and seaborn theme retained."""
from pathlib import Path
import json,sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BLUE,LIGHT_BLUE,RED='#2F6690','#7FB3D5','#B23A48'
DAYS=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

def sql_analysis(d,output):
    output=Path(output);output.mkdir(exist_ok=True,parents=True)
    f=d[['source_row','ChargerID','UserID','Location','ChargerType','date_group','hour','dow','month','known_user','Demand']].copy()
    f['connection_hour']=np.floor(f.hour).astype(int);f['year_month']=f.date_group.str[:7]
    chargers=f[['ChargerID','Location','ChargerType']].drop_duplicates()
    assert chargers.ChargerID.is_unique
    db=sqlite3.connect(output/'charging_analysis.sqlite')
    f.drop(columns=['Location','ChargerType']).to_sql('sessions',db,index=False,if_exists='replace')
    chargers.to_sql('chargers',db,index=False,if_exists='replace')
    queries={
        'Q1_scope':'SELECT COUNT(*) sessions,COUNT(DISTINCT ChargerID) chargers,SUM(Demand) total_kWh,AVG(Demand) mean_kWh FROM sessions',
        'Q2_hour':'SELECT connection_hour,COUNT(*) sessions,SUM(Demand) total_kWh,AVG(Demand) mean_kWh FROM sessions GROUP BY connection_hour ORDER BY connection_hour',
        'Q3_weekday':'SELECT dow,COUNT(*) sessions,SUM(Demand) total_kWh,AVG(Demand) mean_kWh FROM sessions GROUP BY dow ORDER BY dow',
        'Q4_month':'SELECT year_month,COUNT(*) sessions,SUM(Demand) total_kWh FROM sessions GROUP BY year_month ORDER BY year_month',
        'Q5_charger':'''WITH a AS (SELECT ChargerID,COUNT(*) sessions,ROUND(SUM(Demand),6) total_kWh FROM sessions GROUP BY ChargerID)
 SELECT *,RANK() OVER(ORDER BY total_kWh DESC) energy_rank FROM a ORDER BY total_kWh DESC,ChargerID''',
        'Q6_identity':'SELECT known_user,COUNT(*) sessions,AVG(Demand) mean_kWh FROM sessions GROUP BY known_user ORDER BY known_user',
        'Q7_hotspot':'SELECT dow,connection_hour,COUNT(*) sessions,SUM(Demand) total_kWh FROM sessions GROUP BY dow,connection_hour ORDER BY dow,connection_hour',
        'Q8_location':'''SELECT c.Location,COUNT(*) sessions,SUM(s.Demand) total_kWh,AVG(s.Demand) mean_kWh
 FROM sessions s JOIN chargers c ON s.ChargerID=c.ChargerID GROUP BY c.Location ORDER BY c.Location''',
        'Q9_share':'''WITH a AS(SELECT ChargerID,SUM(Demand) total_kWh FROM sessions GROUP BY ChargerID)
 SELECT *,100.0*total_kWh/SUM(total_kWh) OVER() pct_energy FROM a ORDER BY ChargerID'''}
    answers={k:pd.read_sql_query(v,db) for k,v in queries.items()}
    def agg(keys):return f.groupby(keys,as_index=False).agg(sessions=('source_row','size'),total_kWh=('Demand','sum'),mean_kWh=('Demand','mean'))
    pd.testing.assert_series_equal(answers['Q2_hour'].sessions,agg('connection_hour').sessions,check_names=False)
    pd.testing.assert_series_equal(answers['Q3_weekday'].total_kWh,agg('dow').total_kWh,check_names=False,rtol=1e-10)
    pd.testing.assert_frame_equal(answers['Q2_hour'],agg('connection_hour'),check_dtype=False,rtol=1e-10)
    pd.testing.assert_frame_equal(answers['Q3_weekday'],agg('dow'),check_dtype=False,rtol=1e-10)
    pd.testing.assert_frame_equal(answers['Q4_month'],agg('year_month').drop(columns='mean_kWh'),check_dtype=False,rtol=1e-10)
    pd.testing.assert_frame_equal(answers['Q6_identity'],f.groupby('known_user',as_index=False).agg(sessions=('source_row','size'),mean_kWh=('Demand','mean')),check_dtype=False,rtol=1e-10)
    pd.testing.assert_frame_equal(answers['Q7_hotspot'],agg(['dow','connection_hour']).drop(columns='mean_kWh'),check_dtype=False,rtol=1e-10)
    pd.testing.assert_frame_equal(answers['Q8_location'],agg('Location'),check_dtype=False,rtol=1e-10)
    scope=answers['Q1_scope'].iloc[0]
    assert int(scope.sessions)==len(f) and int(scope.chargers)==f.ChargerID.nunique()
    assert np.isclose(scope.total_kWh,f.Demand.sum()) and np.isclose(scope.mean_kWh,f.Demand.mean())
    expected=agg('ChargerID').drop(columns='mean_kWh')
    # Canonical precision avoids floating-point summation changing the order/rank of ties.
    expected['total_kWh']=expected.total_kWh.round(6)
    expected['energy_rank']=expected.total_kWh.rank(method='min',ascending=False).astype(int)
    expected=expected.sort_values(['total_kWh','ChargerID'],ascending=[False,True]).reset_index(drop=True)
    pd.testing.assert_frame_equal(answers['Q5_charger'],expected,check_dtype=False,rtol=1e-10)
    shares=agg('ChargerID').drop(columns=['sessions','mean_kWh']);shares['pct_energy']=100*shares.total_kWh/shares.total_kWh.sum()
    pd.testing.assert_frame_equal(answers['Q9_share'],shares,check_dtype=False,rtol=1e-10)
    assert pd.read_sql_query('SELECT COUNT(*) n FROM sessions s JOIN chargers c ON s.ChargerID=c.ChargerID',db).iloc[0,0]==len(f)
    for k,v in answers.items():v.to_csv(output/(k+'.csv'),index=False)
    (output/'queries.sql').write_text('\n\n'.join('-- '+k+'\n'+v+';' for k,v in queries.items()))
    db.close()
    return answers


def save_figure(fig,folder,name):
    folder=Path(folder);folder.mkdir(exist_ok=True,parents=True)
    # Validate encoded image bytes before replacing the previous artifact.
    from io import BytesIO
    from PIL import Image
    fig.tight_layout()
    data=BytesIO();fig.savefig(data,format='png',dpi=150,bbox_inches='tight')
    payload=data.getvalue()
    assert len(payload)>0, name
    with Image.open(BytesIO(payload)) as im:im.verify()
    (folder/(name+'.png')).write_bytes(payload)
    plt.show();plt.close(fig)


def eda(d,answers,folder):
    fig,ax=plt.subplots(figsize=(8,4.5))
    cols=['UserID','ChargerID','StartDatetime','EndDatetime','Demand']
    sns.heatmap(d[cols].sample(min(600,len(d)),random_state=42).isna(),cbar=False,
        yticklabels=False,cmap=['#E8F1FA',RED],ax=ax)
    ax.set(title='Missing-value pattern in a reproducible sample',xlabel='Field',ylabel='Sampled charging sessions')
    save_figure(fig,folder,'01_missing')
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    sns.histplot(d.Demand,bins=40,kde=True,ax=axes[0],color=BLUE)
    axes[0].axvline(d.Demand.median(),color=RED,linestyle='--',label='Median');axes[0].legend()
    axes[0].set(title='Distribution of delivered energy per session',xlabel='Delivered energy (kWh)',ylabel='Sessions')
    sns.boxplot(x=d.Demand,color=LIGHT_BLUE,ax=axes[1]);axes[1].set(title='Delivered-energy spread',xlabel='Delivered energy (kWh)')
    save_figure(fig,folder,'02_distribution')
    h=answers['Q2_hour'];fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True)
    sns.barplot(data=h,x='connection_hour',y='sessions',color=LIGHT_BLUE,ax=axes[0]);axes[0].set(title='Charging-session arrivals by local connection hour',xlabel='',ylabel='Sessions')
    sns.barplot(data=h,x='connection_hour',y='total_kWh',color=BLUE,ax=axes[1]);axes[1].set(title='Session energy associated with each arrival hour',xlabel='Local connection hour',ylabel='Total session energy (kWh)')
    save_figure(fig,folder,'03_hour')
    w=answers['Q3_weekday'].copy();w['day_name']=w.dow.map(dict(enumerate(DAYS)))
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,y,c,title in [(axes[0],'total_kWh',BLUE,'Total session energy by connection weekday'),(axes[1],'mean_kWh',LIGHT_BLUE,'Mean session energy by connection weekday')]:
        sns.barplot(data=w,x='day_name',y=y,color=c,ax=ax);ax.tick_params(axis='x',rotation=35);ax.set(title=title,xlabel='Day of week',ylabel='Energy (kWh)')
    save_figure(fig,folder,'04_weekday')
    m=answers['Q4_month'];fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
    axes[0].plot(m.year_month,m.sessions,marker='o',color=BLUE);axes[0].set(title='Monthly charging-session count',ylabel='Sessions')
    axes[1].plot(m.year_month,m.total_kWh,marker='o',color=RED);axes[1].set(title='Monthly session energy',ylabel='Energy (kWh)',xlabel='Arrival month');axes[1].tick_params(axis='x',rotation=40)
    save_figure(fig,folder,'05_monthly')
    c=answers['Q5_charger'].head(15).sort_values('total_kWh');fig,ax=plt.subplots(figsize=(10,7))
    sns.barplot(data=c,y='ChargerID',x='total_kWh',color=BLUE,ax=ax);ax.set(title='Top chargers by total session energy',xlabel='Total energy (kWh)',ylabel='Charger ID')
    save_figure(fig,folder,'06_charger')
    heat=answers['Q7_hotspot'].pivot(index='dow',columns='connection_hour',values='total_kWh').reindex(range(7)).fillna(0)
    heat.index=DAYS;fig,ax=plt.subplots(figsize=(14,5.5));sns.heatmap(heat,cmap='Blues',linewidths=.2,ax=ax)
    ax.set(title='Session energy by connection weekday and local hour',xlabel='Local connection hour',ylabel='Day of week');save_figure(fig,folder,'07_heatmap')
    i=answers['Q6_identity'].copy();i['identity']=i.known_user.map({0:'Anonymous',1:'Identified'})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,y,c in [(axes[0],'sessions',BLUE),(axes[1],'mean_kWh',LIGHT_BLUE)]:
        sns.barplot(data=i,x='identity',y=y,color=c,ax=ax);ax.set(xlabel='Arrival identity status',ylabel='Sessions' if y=='sessions' else 'Mean energy (kWh)')
    axes[0].set_title('Sessions by identity availability');axes[1].set_title('Mean energy by identity availability');save_figure(fig,folder,'08_identity')
    fig,ax=plt.subplots(figsize=(7,5.5));sns.heatmap(d[['hour','dow','Demand']].corr(method='spearman'),annot=True,fmt='.2f',cmap='coolwarm',center=0,vmin=-1,vmax=1,ax=ax)
    ax.set_title('Spearman correlation of calendar variables and energy');save_figure(fig,folder,'09_correlation')
    fig,ax=plt.subplots(figsize=(10,5));sample=d.sample(min(16000,len(d)),random_state=42)
    for value,label,c in [(0,'Weekday',BLUE),(1,'Weekend',RED)]:
        s=sample[sample.is_weekend==value];ax.scatter(s.hour,s.Demand,s=8,alpha=.2,label=label,color=c)
    ax.set(title='Arrival hour and session energy',xlabel='Local arrival hour',ylabel='Delivered energy (kWh)',xlim=(0,24));ax.legend();save_figure(fig,folder,'10_scatter')


def model_figures(output):
    output=Path(output);folder=output/'figures';m=output/'models'
    cv=pd.read_csv(m/'comparison_cv.csv');selected=json.loads((output/'manifest.json').read_text())['selected_model']
    fig,ax=plt.subplots(figsize=(11,6));v=cv.sort_values('cv_R2')
    ax.barh(v.model,v.cv_R2,xerr=v.cv_R2_sd,color=[RED if n==selected else BLUE for n in v.model],capsize=3)
    ax.set(title='Regression comparison on temporal CV',xlabel='Mean CV R² with fold standard deviation');save_figure(fig,folder,'12_model_cv')
    a=pd.read_csv(m/'feature_ablation.csv');fig,ax=plt.subplots(figsize=(10,5));ax.barh(a.feature_set,a.cv_R2,color=BLUE)
    ax.set(title='Feature ablation under fixed LightGBM',xlabel='Mean temporal CV R²');save_figure(fig,folder,'11_ablation')
    p=pd.read_csv(m/'personalized_test_predictions.csv');y=p.Demand.to_numpy();pred=p[selected].to_numpy();res=y-pred
    fig,ax=plt.subplots(figsize=(7,6));ax.scatter(y,pred,s=12,alpha=.25,color=BLUE)
    lim=[min(y.min(),pred.min()),max(y.max(),pred.max())];ax.plot(lim,lim,'--',color=RED,label='Perfect prediction')
    ax.set(title='Predicted versus actual session energy',xlabel='Actual energy (kWh)',ylabel='Predicted energy (kWh)');ax.legend();save_figure(fig,folder,'13_predicted_actual')
    fig,ax=plt.subplots(figsize=(8,5));ax.scatter(pred,res,s=12,alpha=.25,color=BLUE);ax.axhline(0,color=RED,linestyle='--')
    ax.set(title='Residuals versus predicted energy',xlabel='Predicted energy (kWh)',ylabel='Actual minus predicted energy (kWh)');save_figure(fig,folder,'14_residuals')
    fig,ax=plt.subplots(figsize=(8,5));sns.histplot(res,bins=35,kde=True,ax=ax,color=BLUE);ax.axvline(0,color=RED,linestyle='--')
    ax.set(title='Residual distribution',xlabel='Actual minus predicted energy (kWh)',ylabel='Test sessions');save_figure(fig,folder,'15_residual_histogram')
    p['year_month']=p.date_group.str[:7];p['absolute_error']=np.abs(res);p['squared_error']=res**2;p['residual']=res
    p['is_weekend']=(pd.to_datetime(p.date_group).dt.dayofweek>=5).astype(int)
    for group in ['year_month','is_weekend','ChargerID']:
        t=p.groupby(group).agg(n=('Demand','size'),MAE=('absolute_error','mean'),RMSE=('squared_error',lambda s:np.sqrt(s.mean())),bias=('residual','mean')).reset_index()
        t.to_csv(m/f'error_by_{group}.csv',index=False)
    month=pd.read_csv(m/'error_by_year_month.csv');fig,ax=plt.subplots(figsize=(9,5));ax.bar(month.year_month,month.MAE,color=BLUE)
    for i,r in enumerate(month.itertuples()):ax.text(i,r.MAE+.05,f'n={r.n}',ha='center',fontsize=9)
    ax.set(title='Selected model error across test months',xlabel='Test arrival month',ylabel='MAE (kWh)',ylim=(0,month.MAE.max()*1.2));save_figure(fig,folder,'16_error_by_month')
    return p
