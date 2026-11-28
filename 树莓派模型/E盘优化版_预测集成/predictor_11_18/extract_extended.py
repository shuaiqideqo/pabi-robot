"""Read fixed stable windows, derive gain-reduced diagnostic features."""
import argparse,hashlib,io,json,sqlite3
from contextlib import closing
from pathlib import Path
import numpy as np
import soundfile as sf
P=Path(__file__).resolve().parent

def features(y,fs=2000000):
    n=len(y);y=y-y.mean();w=np.hanning(n);z=np.fft.rfft(y*w);p=np.abs(z)**2;p[1:-1]*=2
    f=np.fft.rfftfreq(n,1/fs);b=(f>=108000)&(f<112000);f0=float(f[np.flatnonzero(b)[np.argmax(p[b])]])
    narrow=p[b].sum();den=n*np.sum(w*w)
    energies=[];floors=[]
    for order,halfwidth in [(1,200),(2,400)]:
        center=order*f0;line=np.abs(f-center)<=halfwidth
        local=(np.abs(f-center)>=1200)&(np.abs(f-center)<4000)
        floor=float(np.median(p[local]));energies.append(max(float(p[line].sum()-floor*line.sum()),1e-30));floors.append(floor*line.sum())
    line50=float(p[np.abs(f-f0)<=50].sum()/max(narrow,1e-30))
    harmonic_ratio=energies[1]/energies[0]
    h2_snr=energies[1]/max(floors[1],1e-30)
    noisebands={}
    for lo,hi in [(20,50),(50,90),(90,108),(112,150),(150,200),(200,250),(250,300)]:
        mask=(f>=lo*1000)&(f<hi*1000)&(np.abs(f-2*f0)>=2000)
        noisebands[f'{lo}_{hi}']=float(p[mask].sum())
    offsum=sum(noisebands.values())
    # Complex fundamental envelope, trim FFT edges before block aggregation.
    raw=np.fft.rfft(y);analytic=np.zeros(n,complex);mask=np.abs(f-f0)<2000
    idx=np.flatnonzero(mask);analytic[idx]=2*raw[idx]
    envelope=np.abs(np.fft.ifft(analytic))[n//20:-n//20]
    envelope=envelope[:len(envelope)//1000*1000].reshape(-1,1000).mean(axis=1)
    envcv=float(envelope.std()/max(envelope.mean(),1e-20))
    # Relative harmonic phase coupling across twenty 5 ms blocks.
    t=np.arange(n)/fs;osc=np.exp(-2j*np.pi*f0*t)
    c1=(y*osc).reshape(20,-1).sum(axis=1);c2=(y*osc*osc).reshape(20,-1).sum(axis=1)
    coupled=np.sum(c2*np.conj(c1)**2)
    bic=float(abs(coupled)/max(np.sqrt(np.sum(abs(c2)**2)*np.sum(abs(c1)**4)),1e-30))
    phase=float(np.angle(coupled))
    return dict(log_tone=np.log10(max(np.sqrt(energies[0]/den),1e-9)),log_h2_ratio=np.log10(max(harmonic_ratio,1e-12)),
                h2_snr_db=float(10*np.log10(max(h2_snr,1e-12))),line_fraction=line50,env_cv=envcv,
                harmonic_coherence=bic,harmonic_cos=float(np.cos(phase)*bic),harmonic_sin=float(np.sin(phase)*bic),
                noise_high_low=float(np.log10(max(noisebands['150_200']+noisebands['200_250']+noisebands['250_300'],1e-30)/max(noisebands['20_50']+noisebands['50_90'],1e-30))),
                noise_mid_low=float(np.log10(max(noisebands['90_108']+noisebands['112_150'],1e-30)/max(noisebands['20_50']+noisebands['50_90'],1e-30))),
                noise_low_fraction=float(noisebands['20_50']/max(offsum,1e-30)),peak_hz=f0)

def extract(path,ids):
    from waveforms import decode_samples
    selected=set(ids);values=[]
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db:
        rows=db.execute('SELECT SetID,SampleRate,Samples,TR_mV,DataFormat,Data FROM view_tr_data ORDER BY Time')
        for ident,fs,n,scale,fmt,blob in rows:
            if ident not in selected:continue
            if fs!=2000000 or n!=200000 or fmt not in (0, 2):
                raise ValueError('Unexpected waveform metadata')
            x=decode_samples(blob, int(fmt), int(n))
            y=x.astype(float)*scale*1000
            values.append(dict(id=ident,**features(y,fs)))
    assert len(values)==len(selected)
    aggregate={k:float(np.median([v[k] for v in values])) for k in values[0] if k!='id'}
    aggregate['harmonic_detected_fraction']=float(np.mean([v['h2_snr_db']>6 for v in values]))
    return aggregate,values

def main():
    a=argparse.ArgumentParser();a.add_argument('--previous',type=Path,required=True);a.add_argument('--source',type=Path,required=True);args=a.parse_args()
    old=json.loads((args.previous/'training_features.json').read_text(encoding='utf-8'))
    provenance=json.loads((args.previous/'provenance.json').read_text(encoding='utf-8'))
    for n,exts in provenance['manifest'].items():
        for ext,m in exts.items():
            p=args.source/(n+'.'+ext);s=p.stat()
            assert s.st_size==m['bytes'] and s.st_mtime_ns==m['mtime_ns'],('SOURCE_CHANGED',p)
    extended=[];detail={}
    for r in old:
        if not r['main']:extended.append(r);continue
        aggregate,rows=extract((args.source/(r['name']+'.tradb')).resolve(),r['main']['ids'])
        r['extended']=aggregate;extended.append(r);detail[r['name']]=rows
        print('EXTRACTED',r['name'],len(rows),json.dumps({k:round(aggregate[k],4) for k in ['log_h2_ratio','harmonic_detected_fraction','env_cv','noise_high_low']},ensure_ascii=True),flush=True)
    (P/'extended_training_features.json').write_text(json.dumps(extended,ensure_ascii=False,indent=2),encoding='utf-8')
    (P/'extended_waveform_features.json').write_text(json.dumps(detail,ensure_ascii=False),encoding='utf-8')
    provenance['extended_feature_method']='fixed stable window spectral/harmonic/envelope diagnostics'
    provenance['extractor_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (P/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    print('COMPLETE',len(extended))

if __name__=='__main__':main()
