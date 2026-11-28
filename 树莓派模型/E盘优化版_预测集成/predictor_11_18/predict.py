import argparse,json
from pathlib import Path
from waveforms import read_pair
from model_core import decision,predict
P=Path(__file__).resolve().parent

def record_from_file(path):
    # read_pair 一次读库：选段 + 仅对选中窗算扩展特征（不再二次 extract 扫 tradb）
    return read_pair(path)

def safe_prediction(model,r,include_experiment=False):
    candidate=decision(model,r)
    a=r.get('main')
    if not model.get('deployment_ready',False) and candidate['status']!='unreliable':
        public=dict(status='uncertain',title='距离不确定',reason='重拟合模型尚未通过跨组精度验证，暂不发布可靠距离。',distanceCm=None)
    else:public=candidate.copy()
    public.update(modelId=model['model_id'],deploymentReady=model.get('deployment_ready',False),experimental=True,
                  stableInterval=None if not a else dict(startSeconds=a['start_s'],endSeconds=a['end_s'],windows=a['n']))
    if include_experiment:
        raw=None if not a or a['clipped_windows'] else predict(model['regressor'],r)
        public['experiment']=dict(warning='未经独立验证的研究结果；误差带不是保证，训练文件复测不能证明精度。',
                                  rawRegressionCm=raw,candidateDecision=candidate,validation=model['validation'])
    return public

def predict_file(path,model_path=None,include_experiment=False):
    model=json.loads(Path(model_path or P/'model_v2.json').read_text(encoding='utf-8'))
    return safe_prediction(model,record_from_file(path),include_experiment)

def main():
    p=argparse.ArgumentParser();p.add_argument('file',type=Path);p.add_argument('--model',type=Path,default=P/'model_v2.json');p.add_argument('--experimental-details',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    try:r=predict_file(a.file,a.model,a.experimental_details)
    except (ValueError,OSError) as exc:r=dict(status='error',title='文件无法分析',reason=str(exc),distanceCm=None)
    text=json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)
    if a.output:a.output.write_text(text,encoding='utf-8')
    print(text)
    if r['status']=='error':raise SystemExit(2)

if __name__=='__main__':main()
