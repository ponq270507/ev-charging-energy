"""Arrival-time features from a supplied, completion-gated history pool.

A development-only pool gives frozen history. A pool containing evaluation
sessions gives online replay: each label remains inaccessible until end + buffer.
"""
from pathlib import Path
from collections import defaultdict, deque
import hashlib
import json
import numpy as np
import pandas as pd

CAT = ['ChargerID', 'UserID', 'ChargerCompany', 'Location', 'ChargerType']
CALENDAR = ['hour', 'dow', 'month', 'is_weekend', 'days_since_start',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
            'known_user']
TARGET = 'Demand'
LABEL_DELAY_SECONDS = 60


def load_and_audit(path, output=None):
    path=Path(path)
    d=pd.read_csv(path)
    original=list(d.columns)
    d.insert(0,'source_row',np.arange(len(d))+2)
    for c in ['StartDatetime','EndDatetime']:
        d[c]=pd.to_datetime(d[c],errors='coerce')
    for c in ['Demand','Duration']:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    elapsed=(d.EndDatetime-d.StartDatetime).dt.total_seconds()/60
    flags=pd.DataFrame({
        'exact_duplicate':d.duplicated(subset=original),
        'invalid_timestamp':d.StartDatetime.isna()|d.EndDatetime.isna(),
        'nonpositive_elapsed':elapsed<=0,
        'invalid_energy':~np.isfinite(d.Demand)|(d.Demand<=0),
        'invalid_duration':~np.isfinite(d.Duration)|(d.Duration<=0),
        'duration_timestamp_disagreement_over_2min':(d.Duration-elapsed).abs()>2,
    })
    d['exclusion_reason']=flags.apply(lambda r:';'.join(r.index[r]),axis=1)
    d['elapsed_minutes_audit']=elapsed
    rejected=d[flags.any(axis=1)].copy()
    clean=d[~flags.any(axis=1)].copy().sort_values(['StartDatetime','source_row']).reset_index(drop=True)
    for c in CAT:
        clean[c]=clean[c].astype(str)
    # The anonymous ID is not one person and never receives personal history.
    clean['known_user']=(clean.UserID!='0').astype(int)
    clean['hour']=clean.StartDatetime.dt.hour+clean.StartDatetime.dt.minute/60+clean.StartDatetime.dt.second/3600
    clean['dow']=clean.StartDatetime.dt.dayofweek
    clean['month']=clean.StartDatetime.dt.month
    clean['is_weekend']=(clean.dow>=5).astype(int)
    clean['days_since_start']=(clean.StartDatetime-clean.StartDatetime.min()).dt.total_seconds()/86400
    clean['date_group']=clean.StartDatetime.dt.strftime('%Y-%m-%d')
    clean['label_available_at']=clean.EndDatetime+pd.Timedelta(seconds=LABEL_DELAY_SECONDS)
    for col,period,offset in [('hour',24,0),('dow',7,0),('month',12,1)]:
        clean[col+'_sin']=np.sin(2*np.pi*(clean[col]-offset)/period)
        clean[col+'_cos']=np.cos(2*np.pi*(clean[col]-offset)/period)
    audit={'source_url':'https://doi.org/10.6084/m9.figshare.22495141.v1',
        'paper_url':'https://doi.org/10.1038/s41597-024-02942-9',
        'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'raw_rows':len(d),'clean_rows':len(clean),'quarantine_rows':len(rejected),
        'overlapping_rule_counts':{c:int(flags[c].sum()) for c in flags},
        'energy_unit':'kWh (source Table 7)','duration_unit':'minutes of connection (source Table 7)',
        'time_convention':'Recorded South Korean civil times; no invented UTC conversion.',
        'duration_tolerance_minutes':2,'label_availability_buffer_seconds':LABEL_DELAY_SECONDS,
        'user_zero':'Anonymous/non-member group; never pool personal history for ID 0.',
        'no_target_capping':True,'columns_actually_supplied':original}
    if output is not None:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        clean.to_csv(output/'ChargingRecords_clean_v2.csv',index=False)
        rejected.to_csv(output/'ChargingRecords_quarantine_v2.csv',index=False)
        (output/'source_audit_v2.json').write_text(json.dumps(audit,indent=2))
    return clean,rejected,audit


def chronological_split(d, fraction=.8):
    dates=np.sort(d.date_group.unique())
    cutoff=dates[int(len(dates)*fraction)]
    test=d[d.date_group>=cutoff].copy().reset_index(drop=True)
    dev=d[(d.date_group<cutoff)&(d.label_available_at<pd.Timestamp(cutoff))].copy().reset_index(drop=True)
    purged=d[(d.date_group<cutoff)&(d.label_available_at>=pd.Timestamp(cutoff))].copy()
    assert dev.label_available_at.max()<test.StartDatetime.min()
    return dev,test,purged,cutoff


class Stats:
    def __init__(self):
        self.n=0;self.total=0.;self.squares=0.;self.durations=0.;self.tail=deque(maxlen=10);self.duration_tail=deque(maxlen=10);self.last=None;self.first=None;self.maximum=0.;self.minimum=np.inf;self.cache=None
    def add(self,y,duration,t):
        self.n+=1;self.total+=y;self.squares+=y*y;self.durations+=duration
        if self.first is None:self.first=t
        self.maximum=max(self.maximum,y);self.minimum=min(self.minimum,y)
        self.tail.append(y);self.duration_tail.append(duration);self.last=t;self.cache=None
    def features(self,t):
        if not self.n:return [0.]+[np.nan]*19
        if self.cache is None:
            mean=self.total/self.n
            tail=np.array(self.tail);duration=np.array(self.duration_tail)
            power=np.divide(tail,duration,out=np.full(len(tail),np.nan),where=duration>0)
            p=power[np.isfinite(power)];p3=power[-3:];p3=p3[np.isfinite(p3)]
            span=(self.last-self.first)/86400
            cycle=span/(self.n-1) if self.n>1 and span>0 else np.nan
            self.cache=[np.log1p(self.n),mean,np.sqrt(max(0.,self.squares/self.n-mean*mean)),
                    tail[-1],tail[-3:].mean(),tail.mean(),np.median(tail),
                    self.durations/self.n,duration[-1],duration[-3:].mean(),np.median(duration),
                    np.median(p) if len(p) else np.nan,p3.mean() if len(p3) else np.nan,
                    self.maximum,self.minimum,np.log1p(span),cycle]
        gap=(t-self.last)/86400
        ratio=gap/self.cache[-1] if np.isfinite(self.cache[-1]) and self.cache[-1]>0 else np.nan
        expected=min(self.maximum,self.cache[1]*ratio) if np.isfinite(ratio) else np.nan
        return self.cache+[gap,ratio,expected]


STAT_NAMES=['log_count','energy_mean','energy_std','energy_last','energy_last3_mean',
            'energy_last10_mean','energy_last10_median','duration_mean','duration_last',
            'duration_last3_mean','duration_last10_median','power_last10_median',
            'power_last3_mean','historical_max_energy','historical_min_energy','log_history_span_days',
            'mean_cycle_days','days_since_label','relative_cycle_gap','gap_energy_proxy']
HISTORY=[f'{entity}_{name}' for entity in ['global','charger','user','charger_weekday','user_weekday','user_charger'] for name in STAT_NAMES]


def build_features(query, history_pool, use_history=True, use_identity=True):
    """For training use query=pool; labels enter the state strictly after completion.

    Pass only development rows for a frozen-history diagnostic. For online replay,
    include the event stream; each query still sees only strictly earlier label
    availability events. Model fitting is separate from feature-store updates.
    """
    query=query.copy()
    cats=CAT if use_identity else [c for c in CAT if c!='UserID']
    X=query[CALENDAR+cats].copy()
    if not use_history:return X,cats
    events=history_pool.sort_values(['label_available_at','source_row'])
    ev_time=events.label_available_at.astype('int64').to_numpy()/1e9
    q_time=query.StartDatetime.astype('int64').to_numpy()/1e9
    ev_y=events.Demand.to_numpy(float)
    ev_d=events.Duration.to_numpy(float)/60
    ev_s=events.ChargerID.astype(str).to_numpy()
    ev_u=events.UserID.astype(str).to_numpy()
    ev_w=events.dow.to_numpy()
    q_s=query.ChargerID.astype(str).to_numpy();q_u=query.UserID.astype(str).to_numpy();q_w=query.dow.to_numpy()
    global_stats=Stats();chargers=defaultdict(Stats);users=defaultdict(Stats);weekday=defaultdict(Stats)
    user_weekday=defaultdict(Stats);user_charger=defaultdict(Stats)
    result=np.empty((len(query),len(HISTORY)),dtype=np.float32)
    latest=np.full(len(query),np.nan)
    ptr=0
    for i in np.argsort(q_time,kind='stable'):
        t=q_time[i]
        while ptr<len(events) and ev_time[ptr]<t:
            y,dur,et=ev_y[ptr],ev_d[ptr],ev_time[ptr]
            global_stats.add(y,dur,et)
            chargers[ev_s[ptr]].add(y,dur,et)
            weekday[(ev_s[ptr],ev_w[ptr])].add(y,dur,et)
            if ev_u[ptr]!='0':
                users[ev_u[ptr]].add(y,dur,et)
                user_weekday[(ev_u[ptr],ev_w[ptr])].add(y,dur,et)
                user_charger[(ev_u[ptr],ev_s[ptr])].add(y,dur,et)
            ptr+=1
        user=users[q_u[i]] if q_u[i]!='0' else Stats()
        uw=user_weekday[(q_u[i],q_w[i])] if q_u[i]!='0' else Stats()
        uc=user_charger[(q_u[i],q_s[i])] if q_u[i]!='0' else Stats()
        result[i]=global_stats.features(t)+chargers[q_s[i]].features(t)+user.features(t)+weekday[(q_s[i],q_w[i])].features(t)+uw.features(t)+uc.features(t)
        if ptr:latest[i]=ev_time[ptr-1]
    assert np.all(np.isnan(latest)|(latest<q_time))
    hist=pd.DataFrame(result,index=X.index,columns=HISTORY)
    X=pd.concat([X,hist],axis=1)
    # audit timestamps are deliberately not predictors
    return X,cats


def audit_feature_causality(frame):
    """A row's own label and later labels must not affect its feature vector."""
    if len(frame)<20:return
    take=frame.iloc[:min(300,len(frame))].copy().reset_index(drop=True)
    x,_=build_features(take,take)
    idx=len(take)//2
    changed=take.copy()
    changed.loc[changed.StartDatetime>=take.loc[idx,'StartDatetime'],'Demand']+=1000000
    z,_=build_features(changed,changed)
    numeric=CALENDAR+HISTORY
    assert np.allclose(x.loc[:idx,numeric],z.loc[:idx,numeric],equal_nan=True)
    # The own/future labels were supplied but remain inaccessible before completion.
    return True


def load_china(path, output=None):
    """Source-defined energy target, with zero-energy failed attempts retained.

    Billing, final termination flags and whole-day measured weather are excluded
    from predictors. User IDs are anonymized persistent account identifiers.
    """
    path=Path(path)
    raw=pd.read_csv(path,encoding='gb18030')
    start=pd.to_datetime(raw['Start Time'].astype(str).str.replace(r'\\t','',regex=True).str.strip(),format='%Y-%m-%d %H:%M:%S',errors='coerce')
    end=pd.to_datetime(raw['End Time'].astype(str).str.replace(r'\\t','',regex=True).str.strip(),format='%Y-%m-%d %H:%M:%S',errors='coerce')
    y=pd.to_numeric(raw['Transaction power/kwh'],errors='coerce')
    minutes=(end-start).dt.total_seconds()/60
    flags=pd.DataFrame({'exact_duplicate':raw.duplicated(),
        'invalid_timestamp':start.isna()|end.isna(),
        'negative_elapsed':minutes<0,
        'positive_energy_with_zero_elapsed':(minutes==0)&(y>0),
        'invalid_energy':~np.isfinite(y)|(y<0)})
    reasons=flags.apply(lambda r:';'.join(r.index[r]),axis=1)
    d=pd.DataFrame({'source_row':np.arange(len(raw))+2,'UserID':raw.UserID.astype(str),
        'ChargerID':raw['Charging Post ID'].astype(str),'ChargerCompany':'StateGrid',
        'Location':raw['Location Information'].astype(str),'District':raw['District Name'].astype(str),
        'ChargerType':'not_provided','StartDatetime':start,'EndDatetime':end,
        'Duration':minutes,'Demand':y,'exclusion_reason':reasons})
    rejected=d[flags.any(axis=1)].copy()
    clean=d[~flags.any(axis=1)].sort_values(['StartDatetime','source_row']).reset_index(drop=True)
    clean['known_user']=(~clean.UserID.isin(['0','nan',''])).astype(int)
    clean['hour']=clean.StartDatetime.dt.hour+clean.StartDatetime.dt.minute/60+clean.StartDatetime.dt.second/3600
    clean['dow']=clean.StartDatetime.dt.dayofweek;clean['month']=clean.StartDatetime.dt.month
    clean['is_weekend']=(clean.dow>=5).astype(int)
    clean['days_since_start']=(clean.StartDatetime-clean.StartDatetime.min()).dt.total_seconds()/86400
    clean['date_group']=clean.StartDatetime.dt.strftime('%Y-%m-%d')
    clean['label_available_at']=clean.EndDatetime+pd.Timedelta(minutes=5)
    for col,period,offset in [('hour',24,0),('dow',7,0),('month',12,1)]:
        clean[col+'_sin']=np.sin(2*np.pi*(clean[col]-offset)/period)
        clean[col+'_cos']=np.cos(2*np.pi*(clean[col]-offset)/period)
    audit={'source_url':'https://doi.org/10.6084/m9.figshare.28182251.v3',
        'paper_url':'https://doi.org/10.1038/s41597-025-04982-1',
        'raw_rows':len(raw),'clean_rows':len(clean),'quarantine_rows':len(rejected),
        'overlapping_rule_counts':{c:int(flags[c].sum()) for c in flags},
        'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'energy_unit':'kWh; source heading says Transaction power/kwh but describes session ENERGY.',
        'zero_energy_rows_retained':int(clean.Demand.eq(0).sum()),
        'postarrival_fields_excluded':['End Time','Payment time','Electricity cost/Yuan','Service charge/Yuan',
            'Transaction Amount/Yuan','Actual Payment/Yuan','end_cause','all is_* termination flags',
            'whole-day measured Temperature, Humidity, Precipitation'],
        'label_availability_buffer_seconds':300,
        'time_convention':'Recorded Jiaxing civil clock time; no invented UTC shift.',
        'no_target_capping':True,'source_columns':list(raw.columns)}
    if output is not None:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        clean.to_csv(output/'Jiaxing_clean.csv',index=False)
        rejected.to_csv(output/'Jiaxing_quarantine.csv',index=False)
        (output/'jiaxing_audit.json').write_text(json.dumps(audit,indent=2))
    return clean,rejected,audit
