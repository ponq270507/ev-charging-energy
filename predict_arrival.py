"""Predict new arrivals without their energy, duration or departure time."""
from pathlib import Path
from decimal import Decimal, InvalidOperation
import hashlib
import numpy as np
import pandas as pd
import joblib
from arrival_history import build_features
from train_arrival import predict_fit


def _identifier(value, *, anonymous_if_missing=False):
    """Canonicalize numeric IDs without losing precision or splitting 0/0.0.

    Missing account IDs are anonymous. Charger IDs must be supplied. Numeric
    identifiers in the supplied schema are nonnegative integers; other nonempty
    labels may represent previously unseen accounts/chargers.
    """
    missing = pd.isna(value)
    text = '' if missing else str(value).strip()
    if missing or text.lower() in {'', 'nan', 'none', 'null', '<na>', 'nat'}:
        if anonymous_if_missing:
            return '0'
        raise ValueError('ChargerID must be nonempty.')
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('Identifiers cannot be booleans.')
    try:
        number = Decimal(text)
    except InvalidOperation:
        return text
    if not number.is_finite() or number < 0 or number != number.to_integral_value():
        raise ValueError('Numeric identifiers must be finite nonnegative integers.')
    return str(int(number))


def _local_times(values, field):
    """Naive timestamps mean recorded Korean local time; aware values are converted."""
    result=[]
    for value in values:
        try:
            stamp=pd.Timestamp(value)
            if pd.isna(stamp):raise ValueError('missing timestamp')
            if stamp.tzinfo is not None:
                stamp=stamp.tz_convert('Asia/Seoul').tz_localize(None)
            result.append(stamp)
        except (TypeError,ValueError,OverflowError) as exc:
            raise ValueError(f'{field} must contain valid timestamps.') from exc
    return pd.Series(result,index=values.index,dtype='datetime64[ns]')


def _category(value):
    if pd.isna(value) or str(value).strip().lower() in {'','nan','none','null','<na>'}:
        return np.nan
    return _identifier(value)


def _append_completed(pool, supplied):
    """Validate and merge a cumulative feed without counting retransmissions twice.

    A charger/account/start/end tuple identifies an event in this source schema.
    Conflicting outcomes or reused source-row IDs are rejected; corrections need
    reconciliation outside this prediction function.
    """
    if supplied is None or len(supplied)==0:return pool
    extra=pd.DataFrame(supplied).copy().reset_index(drop=True)
    required=['ChargerID','UserID','StartDatetime','EndDatetime','Duration','Demand']
    if not set(required)<=set(extra):raise ValueError('Completed history is missing required recorded fields.')
    extra['ChargerID']=extra.ChargerID.map(_identifier)
    extra['UserID']=extra.UserID.map(lambda v:_identifier(v,anonymous_if_missing=True))
    for c in ['ChargerCompany','ChargerType','Location']:
        if c in extra:extra[c]=extra[c].map(_category)
    for c in ['StartDatetime','EndDatetime']:extra[c]=_local_times(extra[c],c)
    for c in ['Demand','Duration']:
        extra[c]=pd.to_numeric(extra[c],errors='coerce')
        if not (np.isfinite(extra[c])&extra[c].gt(0)).all():
            raise ValueError(f'Completed {c} must be finite and positive.')
    elapsed=(extra.EndDatetime-extra.StartDatetime).dt.total_seconds()/60
    if not (elapsed.gt(0)&extra.Duration.sub(elapsed).abs().le(2)).all():
        raise ValueError('Completed history requires EndDatetime > StartDatetime and duration agreement within two minutes.')
    keys=['ChargerID','UserID','StartDatetime','EndDatetime']
    def event(row):return tuple(row[c] for c in keys)
    existing={event(row):row for row in pool.to_dict('records')}
    source_keys={int(row['source_row']):event(row) for row in pool.to_dict('records')}
    accepted=[]
    for row in extra.to_dict('records'):
        key=event(row)
        previous=existing.get(key)
        if previous is not None:
            if not all(np.isclose(row[c],previous[c],rtol=1e-10,atol=1e-9) for c in ['Duration','Demand']):
                raise ValueError('Conflicting outcomes for the same completed event.')
        raw_id=row.get('source_row')
        if raw_id is not None and not pd.isna(raw_id):
            sid=_identifier(raw_id)
            try:sid=int(sid)
            except ValueError as exc:raise ValueError('source_row must be a nonnegative integer when supplied.') from exc
        else:
            # Stable tie order under feed reordering; IDs do not become predictors.
            token='|'.join(str(v) for v in key).encode()
            sid=-1-int.from_bytes(hashlib.sha256(token).digest()[:7],'big')
        if sid in source_keys and source_keys[sid]!=key:
            raise ValueError('source_row is reused for a different completed event.')
        if previous is not None:continue
        row['source_row']=sid
        existing[key]=row;source_keys[sid]=key;accepted.append(row)
    if not accepted:return pool
    extra=pd.DataFrame(accepted)
    extra['label_available_at']=extra.EndDatetime+pd.Timedelta(seconds=60)
    extra['dow']=extra.StartDatetime.dt.dayofweek
    return pd.concat([pool,extra],ignore_index=True)


def predict_sessions(bundle_path,arrivals,completed_since_training=None):
    """Predict at/after the model's fitting cutoff; never refit model weights.

    Completion-gated online history may update through a validated cumulative
    feed. Outcomes still unavailable at each query remain excluded.
    """
    bundle=joblib.load(bundle_path)
    pool=_append_completed(bundle['history_pool_development'].copy(),completed_since_training)
    q=pd.DataFrame(arrivals).copy().reset_index(drop=True)
    for c in ['ChargerID','UserID','StartDatetime']:
        if c not in q:raise ValueError(f'Arrival input requires {c}.')
    q['ChargerID']=q.ChargerID.map(_identifier)
    q['UserID']=q.UserID.map(lambda v:_identifier(v,anonymous_if_missing=True))
    if q.empty:raise ValueError('At least one arrival is required.')
    q.StartDatetime=_local_times(q.StartDatetime,'StartDatetime')
    if q.StartDatetime.lt(pd.Timestamp(bundle['protocol']['test_start'])).any():
        raise ValueError('Arrival precedes the saved model fitting cutoff; use temporal evaluation code for historical backtests.')
    metadata=['ChargerCompany','ChargerType','Location']
    catalog=pool.sort_values(['StartDatetime','source_row']).copy()
    catalog[metadata]=catalog.groupby('ChargerID',sort=False)[metadata].ffill()
    looked_up=pd.merge_asof(q[['ChargerID','StartDatetime']].assign(_position=np.arange(len(q))).sort_values('StartDatetime'),
        catalog[['ChargerID','StartDatetime']+metadata],on='StartDatetime',by='ChargerID',direction='backward')
    looked_up=looked_up.sort_values('_position').reset_index(drop=True)
    for c in metadata:
        supplied=q[c].map(_category) if c in q else pd.Series(np.nan,index=q.index)
        q[c]=supplied.fillna(looked_up[c]).fillna('unknown').astype(str)
    q['known_user']=q.UserID.ne('0').astype(int)
    q['hour']=q.StartDatetime.dt.hour+q.StartDatetime.dt.minute/60+q.StartDatetime.dt.second/3600
    q['dow']=q.StartDatetime.dt.dayofweek;q['month']=q.StartDatetime.dt.month
    q['is_weekend']=(q.dow>=5).astype(int)
    origin=bundle['history_pool_development'].StartDatetime.min()
    q['days_since_start']=(q.StartDatetime-origin).dt.total_seconds()/86400
    for col,period,offset in [('hour',24,0),('dow',7,0),('month',12,1)]:
        q[col+'_sin']=np.sin(2*np.pi*(q[col]-offset)/period)
        q[col+'_cos']=np.cos(2*np.pi*(q[col]-offset)/period)
    X,_=build_features(q,pool)
    out=q[['ChargerID','UserID','StartDatetime']].copy()
    pred=np.empty(len(q));known=q.known_user.eq(1).to_numpy()
    if known.any():pred[known]=predict_fit(bundle['personalized'],X.loc[known])
    if (~known).any():pred[~known]=predict_fit(bundle['anonymous_fallback'],X.loc[~known])
    out['predicted_kWh']=pred
    out['model_route']=np.where(known,'personalized','anonymous fallback')
    return out
