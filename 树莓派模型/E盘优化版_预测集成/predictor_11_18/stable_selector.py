"""Distance-blind, fixed-rule stable interval analysis of all five sessions."""
import json, sys
from pathlib import Path
import numpy as np

FEATURES=['rms_uv','narrow_rms_uv','offband_rms_uv','narrow_fraction','peak_khz','crest','shape_factor','entropy']
RULES=dict(edge_guard_s=1.0,min_duration_s=3.0,min_windows=15,max_start_gap_s=0.45,
           window_samples=200000,window_duration_s=0.1,main_relative_p90_p10_limit=0.20,
           relative_first_last_third_drift_limit=0.10,narrow_fraction_p90_p10_limit=0.10,
           maximum_relative_single_window_deviation=0.40,
           rolling_median_windows=5,maximum_relative_rolling_median_deviation=0.10,
           sensitivity_relative_widths=[0.15,0.20,0.30],
           selection='longest elapsed interval; then most windows; then smallest normalized instability; then earliest',
           clipping='retain and flag; stable clipped amplitude is not evidence of mechanical stability')

def stats(v):
    a=np.asarray(v,dtype=float)
    return dict(zip(['min','p10','median','p90','max'],map(float,np.percentile(a,[0,10,50,90,100]))))

def candidate_metrics(a):
    # Columns: total RMS, narrowband RMS, band energy fraction.
    p=np.percentile(a,[10,50,90],axis=0); med=p[1,:2]
    if np.any(med<=0):return None
    width=float(np.max((p[2,:2]-p[0,:2])/med))
    third=max(3,len(a)//3)
    drift=float(np.max(np.abs(np.median(a[-third:,:2],axis=0)-np.median(a[:third,:2],axis=0))/med))
    spike=float(np.max(np.abs(a[:,:2]/med-1)))
    fraction=float(p[2,2]-p[0,2])
    # One scalar permits predeclared strict/main/loose comparisons without retuning.
    local=np.median(np.lib.stride_tricks.sliding_window_view(a[:,:2],5,axis=0),axis=2)
    local_deviation=float(np.max(np.abs(local/med-1)))
    need=max(width,2*drift,2*fraction,spike/2,2*local_deviation)
    return dict(width=width,drift=drift,fraction_width=fraction,spike=spike,local_deviation=local_deviation,required_width=need)

def summarize_selection(c,records,t0):
    if c is None:return None
    rr=records[c['i']:c['j']]
    start=rr[0]['time']-t0;end=rr[-1]['time']-t0+0.1
    return dict(start_s=start,end_s=end,duration_s=end-start,n=len(rr),ids=[r['id'] for r in rr],
                clipped_windows=sum(r['clip_samples']>0 for r in rr),
                criteria=c['metrics'],features={k:stats([r[k] for r in rr]) for k in FEATURES})

def inspect(records):
    t0=records[0]['time']; end=max(r['time']+r['n']/2000000 for r in records)
    full=[r for r in records if r['n']==200000]
    eligible=[r for r in full if r['time']>=t0+1 and r['time']+0.1<=end-1]
    blocks=[];start=0
    for i in range(1,len(eligible)):
        if eligible[i]['time']-eligible[i-1]['time']>0.45:
            blocks.append((start,i));start=i
    if eligible:blocks.append((start,len(eligible)))
    a=np.array([[r[k] for k in ['rms_uv','narrow_rms_uv','narrow_fraction']] for r in eligible])
    candidates=[]
    for b,e in blocks:
        for i in range(b,e):
            for j in range(i+15,e+1):
                duration=eligible[j-1]['time']-eligible[i]['time']+0.1
                if duration<3-1e-8:continue
                metrics=candidate_metrics(a[i:j])
                if metrics and metrics['required_width']<=0.30+1e-10:
                    candidates.append(dict(i=i,j=j,duration=duration,metrics=metrics))
    def rank(c):return (-round(c['duration'],6),-(c['j']-c['i']),c['metrics']['required_width'],c['i'])
    candidates.sort(key=rank)
    results={}
    for label,tol in [('strict',0.15),('main',0.20),('loose',0.30)]:
        selected=next((c for c in candidates if c['metrics']['required_width']<=tol+1e-10),None)
        results[label]=summarize_selection(selected,eligible,t0)
    main=next((c for c in candidates if c['metrics']['required_width']<=0.20+1e-10),None)
    alternate=next((c for c in candidates if c['metrics']['required_width']<=0.20+1e-10 and main and (c['j']<=main['i'] or c['i']>=main['j'])),None)
    results.update(alternate_disjoint=summarize_selection(alternate,eligible,t0),total_records=len(records),
                   full_records=len(full),eligible_records=len(eligible),file_duration_s=end-t0,
                   all_full_features={k:stats([r[k] for r in full]) for k in FEATURES})
    return results
