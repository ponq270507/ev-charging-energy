from pathlib import Path
import json,hashlib,sys
import numpy as np,pandas as pd
p=Path(__file__).resolve().parent
x=pd.read_excel(p/'data/ChargingRecords.xlsx');c=pd.read_csv(p/'data/ChargingRecords_original.csv');cols=list(x.columns)
assert cols==list(c.columns) and x.shape==c.shape==(72856,13)
def canonical(frame):
 z=frame[cols].copy()
 for col in cols:
  if col in ['StartDatetime','EndDatetime']:
   z[col]=pd.to_datetime(z[col],errors='raise',format='mixed').dt.strftime('%Y-%m-%d %H:%M:%S')
  elif col in ['StartDay','EndDay']:
   if pd.api.types.is_numeric_dtype(z[col]):z[col]=pd.to_datetime(z[col],unit='D',origin='1899-12-30').dt.strftime('%Y-%m-%d')
   else:z[col]=pd.to_datetime(z[col],format='mixed').dt.strftime('%Y-%m-%d')
  elif col in ['StartTime','EndTime']:
   if pd.api.types.is_numeric_dtype(z[col]):z[col]=np.rint(z[col].to_numpy(float)*86400).astype(int)%86400
   else:z[col]=pd.to_timedelta(z[col].astype(str)).dt.total_seconds().astype(int)
  elif col=='Location':z[col]=z[col].fillna('<NA>').astype(str)
  else:z[col]=pd.to_numeric(z[col],errors='raise').astype(float).round(10)
 return z
xc,cc=canonical(x),canonical(c)
results={col:bool(xc[col].equals(cc[col])) for col in cols};assert all(results.values()),results
sys.path.insert(0,str(p));from arrival_history import load_and_audit
clean,q,a=load_and_audit(p/'data/ChargingRecords_original.csv');qnew=pd.read_csv(p/'data/ChargingRecords_quarantine_supplied.csv')
qa,qb=canonical(q),canonical(qnew)
ha=pd.util.hash_pandas_object(qa,index=False).sort_values().to_numpy();hb=pd.util.hash_pandas_object(qb,index=False).sort_values().to_numpy()
assert len(q)==len(qnew)==3286 and np.array_equal(ha,hb),'Quarantine raw contents differ'
print('Current exclusion reasons:',q.exclusion_reason.value_counts().to_dict())
print('Attached quality_flags:',qnew.quality_flags.value_counts().to_dict())
qa['reason']=q.exclusion_reason.astype(str).str.replace('exact_duplicate','duplicate',regex=False).str.replace('duration_timestamp_disagreement_over_2min','duration_disagreement',regex=False).to_numpy()
qb['reason']=qnew.quality_flags.astype(str).to_numpy()
assert np.array_equal(pd.util.hash_pandas_object(qa,index=False).sort_values().to_numpy(),pd.util.hash_pandas_object(qb,index=False).sort_values().to_numpy())
result={'uploaded_xlsx':'ChargingRecords.xlsx','uploaded_xlsx_sha256':hashlib.sha256((p/'data/ChargingRecords.xlsx').read_bytes()).hexdigest(),'raw_csv_sha256':hashlib.sha256((p/'data/ChargingRecords_original.csv').read_bytes()).hexdigest(),'raw_rows':len(c),'raw_columns':len(cols),'row_order_preserved':True,'per_column_content_matches':results,'conversion_only':'Excel date serial -> ISO dates; Excel time fractions -> seconds-of-day; timestamps interpreted as recorded local times. Source cells unchanged.','quarantine_rows':len(q),'quarantine_raw_content_multiset_matches':True,'external_dataset_added':False}
result['quarantine_flags_match_after_name_mapping']=True
(p/'verification/source_equivalence.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
