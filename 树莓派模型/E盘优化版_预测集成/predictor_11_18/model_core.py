"""Second-round low-complexity, group-aware regression; no filename inference."""
import numpy as np

SETS={
 'amplitude':['log_band'],
 'shape_only':['log_crest','entropy','shape_factor'],
 'noise_context':['log_band','noise_high_low','noise_mid_low'],
 'envelope':['log_band','env_cv','line_fraction'],
 'spectral':['log_tone','noise_high_low','env_cv'],
 'harmonic':['log_band','harmonic_ratio','harmonic_detected_fraction'],
 'shape_context':['log_band','log_crest','noise_high_low'],
 'normalized':['env_cv','noise_high_low','noise_mid_low','harmonic_detected_fraction'],
}

def vector(r,keys):
    f=r['main']['features'];e=r['extended'];v=lambda k:f[k]['median']
    values={'log_band':np.log10(max(v('narrow_rms_uv'),1e-9)),
            'log_crest':np.log(max(v('crest'),1e-9)), 'entropy':v('entropy'),'shape_factor':v('shape_factor'),**e}
    values['harmonic_ratio']=e['log_h2_ratio'] if e['harmonic_detected_fraction']>=.5 else np.nan
    return np.array([values[k] for k in keys],float)

def candidates():
    result=[dict(id=f'{name}_{mode}_10',features=keys,mode=mode,alpha=10.) for name,keys in SETS.items() for mode in ['ridge','within_group']]
    result.append(dict(id='amplitude_ridge_1',features=SETS['amplitude'],mode='ridge',alpha=1.))
    return result

def eligible(records,scope='range'):
    return [r for r in records if r.get('distance_cm') is not None and r['main'] and not r['main']['clipped_windows'] and
            (scope=='boundary' or r['distance_cm'] in [10,20,30,40])]

def target(r,scope):
    return 5. if scope=='boundary' and r['distance_cm']<=10 else float(r['distance_cm'])

def fit(records,c,scope='range'):
    x=np.array([vector(r,c['features']) for r in records]);y=np.array([target(r,scope) for r in records]);groups=np.array([r['group'] for r in records])
    impute=np.array([float(np.median(x[np.isfinite(x[:,j]),j])) if np.any(np.isfinite(x[:,j])) else 0 for j in range(x.shape[1])])
    x=np.where(np.isfinite(x),x,impute);mean=x.mean(axis=0);scale=x.std(axis=0);scale=np.where(scale<1e-9,1.,scale);z=(x-mean)/scale
    yc=y-y.mean();design=z.copy()
    if c['mode']=='within_group':
        for group in sorted(set(groups)):
            mask=groups==group;design[mask]-=design[mask].mean(axis=0);yc[mask]-=yc[mask].mean()
    coef=np.linalg.solve(design.T@design+c['alpha']*np.eye(x.shape[1]),design.T@yc)
    return dict(config=c,scope=scope,impute=impute.tolist(),mean=mean.tolist(),scale=scale.tolist(),coef=coef.tolist(),intercept=float(y.mean()),
                training_names=[r['name'] for r in records],training_groups=sorted(set(map(int,groups))))

def predict(model,r):
    x=vector(r,model['config']['features']);x=np.where(np.isfinite(x),x,np.array(model['impute']))
    return float(((x-np.array(model['mean']))/np.array(model['scale']))@np.array(model['coef'])+model['intercept'])

def cv(records,c,scope='range'):
    rows=[]
    for g in sorted({r['group'] for r in records}):
        tr=[r for r in records if r['group']!=g];te=[r for r in records if r['group']==g];m=fit(tr,c,scope)
        assert g not in m['training_groups']
        for r in te:
            p=predict(m,r);t=target(r,scope);rows.append(dict(name=r['name'],group=g,true_cm=r['distance_cm'],target=t,prediction_cm=p,error_cm=abs(p-t)))
    return rows

def score(rows):
    mae=float(np.mean([r['error_cm'] for r in rows]));worst=max(np.mean([r['error_cm'] for r in rows if r['group']==g]) for g in {r['group'] for r in rows})
    return dict(mae=mae,max_error=max(r['error_cm'] for r in rows),worst_group_mae=float(worst),objective=mae+.25*float(worst))

def select(records,scope='range'):
    ranking=[]
    for c in candidates():
        oof=cv(records,c,scope);ranking.append(dict(config=c,oof=oof,**score(oof)))
    ranking.sort(key=lambda r:(r['objective'],len(r['config']['features']),r['config']['id']))
    return ranking[0],ranking

def pipeline(records):
    rr=eligible(records);br=eligible(records,'boundary')
    chosen,ranking=select(rr);bc,branking=select(br,'boundary')
    radius=float(np.quantile([r['error_cm'] for r in chosen['oof']],.90))
    bradius=float(np.quantile([r['error_cm'] for r in bc['oof']],.90))
    noise=[x for r in records if r.get('distance_cm') is None for x in r.get('noise_fractions',[])]
    model=dict(model_id='1118_16V_0_40_refit_v2',regressor=fit(rr,chosen['config']),boundary=fit(br,bc['config'],'boundary'),
               global_error_radius_cm=radius,boundary_error_radius_cm=bradius,
               noise_fraction_upper=float(np.quantile(noise,.99)+.05) if noise else None,
               training_groups=sorted({r['group'] for r in records}),deployment_ready=False)
    return model,ranking,branking

def decision(model,r):
    d=dict(status='uncertain',title='距离不确定',distanceCm=None,reason='',experimental=True)
    if r['full_records']<15:d.update(status='unreliable',title='无法可靠判断',reason='完整波形数量不足。');return d
    if model['noise_fraction_upper'] is not None and r['peak_band_fraction']<.2 and r['all_full_features']['narrow_fraction']['median']<=model['noise_fraction_upper']:
        d.update(status='unreliable',title='无法可靠判断',reason='目标频段信号不足。');return d
    if not r['main']:d.update(status='unreliable',title='无法可靠判断',reason='没有合格平稳区间。');return d
    if r['main']['clipped_windows']:d.update(status='unreliable',title='无法可靠判断',reason='平稳段包含削顶，不能按幅值判断距离。');return d
    a=r.get('alternate_disjoint')
    if a:
        values=[r['main']['features']['narrow_rms_uv']['median'],a['features']['narrow_rms_uv']['median']]
        if max(values)/max(min(values),1e-9)>1.35:d['reason']='文件内存在幅值不同的平稳平台。';return d
    raw=predict(model['regressor'],r);b=predict(model['boundary'],r);rad=model['global_error_radius_cm'];brad=model['boundary_error_radius_cm']
    d['diagnostics']=dict(rawRegressionCm=raw,empiricalRange=[raw-rad,raw+rad],boundaryScore=b,boundaryEmpiricalRange=[b-brad,b+brad],
                          warning='经验误差带不是有概率保证的置信区间。')
    if b+brad<=10:d.update(status='near',title='近场：约0–10 cm',reason='边界估计与训练组误差带均支持近场。');return d
    if b-brad>40:d.update(status='outside',title='疑似超出40 cm有效预测范围',reason='边界参考支持超范围，41–49 cm尚未验证。');return d
    if b-brad<10 or b+brad>40:d['reason']='经验误差带跨越近场或范围外边界。';return d
    if rad>8 or raw-rad<10 or raw+rad>40:d['reason']='跨组误差或有效范围不足以支持数值输出。';return d
    d.update(status='distance',title=f'预测距离：{raw:.1f} cm',distanceCm=round(raw,1),reason='实验数值估计，待独立验证。');return d
