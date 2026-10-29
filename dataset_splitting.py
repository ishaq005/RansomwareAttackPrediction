"""Stage 3 chronological audit and partitioning. Run Python File in VS Code.
Place weekly_status_prediction_panel.csv alongside this script.
No model is trained; the 2026 test labels must remain untouched in model selection.
"""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parent
INPUT=ROOT/'weekly_status_prediction_panel.csv'
OUT=ROOT/'data'
OUT.mkdir(exist_ok=True)
REQUIRED=['group','week_start','attacks_7d','attacks_30d','attacks_90d','attacks_120d','attacks_365d','days_since_last_post','campaign_status','next_week_status']
LABELS=['attack_week','active','declining','dormant','inactive']
df=pd.read_csv(INPUT)
errors=[]; warnings=[]
missing=[x for x in REQUIRED if x not in df.columns]
if missing: raise ValueError(f'Missing columns: {missing}')
df['week_start']=pd.to_datetime(df['week_start'],errors='coerce',utc=True)
if df.week_start.isna().any(): errors.append('Unparseable observation-week dates')
if df.duplicated(['group','week_start']).any(): errors.append('Duplicate group-week keys')
if not df.week_start.dt.dayofweek.eq(0).all(): errors.append('Non-Monday observation week')
for x in ['attacks_7d','attacks_30d','attacks_90d','attacks_120d','attacks_365d','days_since_last_post']:
    if df[x].isna().any(): errors.append(f'Missing feature: {x}')
    if (df[x]<0).any(): errors.append(f'Negative feature: {x}')
for a,b in [('attacks_7d','attacks_30d'),('attacks_30d','attacks_90d'),('attacks_90d','attacks_120d'),('attacks_120d','attacks_365d')]:
    if (df[a]>df[b]).any(): errors.append(f'Lookback monotonicity violation: {a}>{b}')
if not (df.attacks_7d.gt(0)==df.campaign_status.eq('attack_week')).all(): errors.append('Current status contradicts attacks_7d')
if (~df.campaign_status.isin(LABELS)).any(): errors.append('Invalid current status')
if (~df.next_week_status.dropna().isin(LABELS)).any(): errors.append('Invalid next-week status')
df=df.sort_values(['group','week_start']).reset_index(drop=True)
next_week=df.groupby('group').week_start.shift(-1)
next_label=df.groupby('group').campaign_status.shift(-1)
mask=next_week.notna()
if not (next_week[mask].reset_index(drop=True)==(df.loc[mask,'week_start']+pd.Timedelta(days=7)).reset_index(drop=True)).all(): errors.append('Gaps in group weekly histories')
if not df.loc[mask,'next_week_status'].reset_index(drop=True).equals(next_label[mask].reset_index(drop=True)): errors.append('Next-week target misalignment')
if df.loc[~mask,'next_week_status'].notna().any(): errors.append('Future target exists for final group week')
if errors: raise ValueError('Audit failed: '+ '; '.join(errors))
labelled=df[df.next_week_status.notna()].copy()
labelled['target_week_start']=labelled.week_start+pd.Timedelta(days=7)
labelled['target_year']=labelled.target_week_start.dt.year
labelled['posting_next_week']=labelled.next_week_status.eq('attack_week').astype(int)
labelled['partition']=labelled.target_year.map(lambda y:'train' if 2022<=y<=2024 else 'validation' if y==2025 else 'test' if y==2026 else 'excluded_early')
features=REQUIRED[:-1]+['target_week_start','posting_next_week','next_week_status']
# Preserve the complete data within each split; do not rebalance here.
for name in ['train','validation','test','excluded_early']:
    subset=labelled[labelled.partition.eq(name)].copy()
    subset.to_csv(OUT/f'{name}.csv',index=False,date_format='%Y-%m-%d')
summary=labelled.groupby('partition',sort=False).agg(observations=('group','size'),groups=('group','nunique'),posting_weeks=('posting_next_week','sum'),first_target_week=('target_week_start','min'),last_target_week=('target_week_start','max')).reset_index()
summary['posting_rate_pct']=(100*summary.posting_weeks/summary.observations).round(3)
summary.to_csv(OUT/'partition_summary.csv',index=False,date_format='%Y-%m-%d')
classes=pd.crosstab(labelled.partition,labelled.next_week_status).reindex(columns=LABELS,fill_value=0)
classes.to_csv(OUT/'partition_class_counts.csv')
annual=labelled.groupby('target_year').agg(observations=('group','size'),groups=('group','nunique'),posting_weeks=('posting_next_week','sum')).reset_index()
annual['posting_rate_pct']=(100*annual.posting_weeks/annual.observations).round(3)
annual.to_csv(OUT/'annual_target_year_coverage.csv',index=False)
if (df['week_start'].max() > pd.Timestamp('2026-09-27',tz='UTC')): warnings.append('Some observations exceed assumed 2026-09-27 cutoff')
warnings.extend(['Source collection/publication-time availability cannot be verified from this 10-column panel.', 'Financial-sector classification and duplicate adjudication require incident-level source review.', 'The 2026 test set must not be used for model or threshold selection.', 'No models trained; these are provisional modelling partitions.'])
report={'panel_rows':len(df),'labelled_rows':len(labelled),'unlabelled_final_rows':int(df.next_week_status.isna().sum()),'unique_groups':int(df.group.nunique()),'audit_passed':True,'errors':errors,'warnings':warnings,'partition_counts':summary[['partition','observations','posting_weeks']].to_dict('records')}
(OUT/'audit_report.json').write_text(json.dumps(report,indent=2))
print(summary.to_string(index=False));print('\nStatus counts:\n',classes.to_string());print('\nAudit passed; limitations saved to audit_report.json')
