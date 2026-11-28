"""Read-only paired SQLite waveform import; inference never uses filename labels.

Supports Vallen DataFormat 2 (FLAC, calibration) and 0 (raw int16, Pi/vallenae live).

One DB pass: light features for stable-interval selection, keep int16 of full
windows in memory, then compute extended/harmonic features only for the selected
ids (no second tradb scan / re-decode).
"""
import io, sqlite3
from contextlib import closing
from pathlib import Path
import numpy as np
import soundfile as sf
from stable_selector import inspect
from extract_extended import features

def decode_samples(blob, fmt, n):
    """Decode one TRA blob to int16 samples. fmt: 2=FLAC, 0=uncompressed PCM."""
    if fmt == 2:
        x = sf.read(io.BytesIO(blob), dtype='int16')[0]
    elif fmt == 0:
        x = np.frombuffer(blob, dtype='<i2')
    else:
        raise ValueError(f'不支持的 DataFormat={fmt}（仅支持 0 原始 / 2 FLAC）。')
    if x.ndim != 1:
        x = np.asarray(x).reshape(-1)
    if len(x) != n:
        raise ValueError('波形解码长度与数据库记录不符。')
    return x

def _aggregate_extended(ids, samples_by_id):
    values = []
    for ident in ids:
        x, scale, fs = samples_by_id[ident]
        y = x.astype(float) * scale * 1000
        values.append(dict(id=ident, **features(y, fs)))
    if not values:
        return None
    aggregate = {k: float(np.median([v[k] for v in values])) for k in values[0] if k != 'id'}
    aggregate['harmonic_detected_fraction'] = float(np.mean([v['h2_snr_db'] > 6 for v in values]))
    return aggregate

def read_pair(path):
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() not in ['.pridb', '.tradb']:
        raise ValueError('请选择 .pridb 或 .tradb 文件。')
    pri = path.with_suffix('.pridb')
    tra = path.with_suffix('.tradb')
    if not pri.is_file() or not tra.is_file():
        raise ValueError('缺少同名 .pridb / .tradb 配对文件。')
    for p in [pri, tra]:
        with p.open('rb') as f:
            if f.read(16) != b'SQLite format 3\x00':
                raise ValueError('文件不是有效 SQLite 数据库。')
    samples_by_id = {}
    try:
        with closing(sqlite3.connect(tra.as_uri() + '?mode=ro', uri=True, timeout=3)) as db:
            meta = db.execute(
                'SELECT DISTINCT SampleRate,TR_mV,DataFormat,Chan FROM view_tr_data'
            ).fetchall()
            if not meta:
                raise ValueError('波形数据库为空。')
            if len(meta) != 1 or meta[0][0] != 2000000 or meta[0][2] not in (0, 2):
                raise ValueError(
                    '采样率/数据格式/通道不匹配：需要单通道 2 MHz，DataFormat=0(原始)或2(FLAC)。'
                )
            if abs(float(meta[0][1]) - 0.0017399999778717756) > 1e-8:
                raise ValueError('波形换算比例与建模数据不同，不能直接预测。')
            cursor = db.execute(
                'SELECT SetID,Time,SampleRate,Samples,TR_mV,DataFormat,Data '
                'FROM view_tr_data ORDER BY Time'
            )
            records = []
            for ident, t, fs, n, scale, fmt, blob in cursor:
                if n <= 0 or n > 1000000:
                    raise ValueError('异常波形长度。')
                x = decode_samples(blob, int(fmt), int(n))
                # copy: frombuffer views may alias the sqlite blob buffer
                if n == 200000:
                    samples_by_id[ident] = (np.array(x, dtype=np.int16, copy=True), float(scale), int(fs))
                y = x.astype(float) * float(scale) * 1000
                y -= y.mean()
                rms = float(np.sqrt(np.mean(y * y)))
                r = dict(
                    id=ident, time=t, n=n, rms_uv=rms,
                    clip_samples=int(np.count_nonzero((x == -32768) | (x == 32767))),
                )
                if n == 200000:
                    w = np.hanning(n)
                    power = np.abs(np.fft.rfft(y * w)) ** 2
                    power[1:-1] *= 2
                    freq = np.fft.rfftfreq(n, 1 / fs)
                    band = (freq >= 108000) & (freq < 112000)
                    total = power.sum()
                    pb = power[band].sum()
                    if rms <= 0 or total <= 0:
                        raise ValueError('完整波形无有效能量。')
                    q = power / total
                    q = q[q > 0]
                    den = n * np.sum(w * w)
                    r.update(
                        narrow_rms_uv=float(np.sqrt(pb / den)),
                        offband_rms_uv=float(np.sqrt(max(0, total - pb) / den)),
                        narrow_fraction=float(pb / total),
                        peak_khz=float(freq[np.argmax(power)] / 1000),
                        crest=float(np.max(np.abs(y)) / rms),
                        shape_factor=float(rms / np.mean(np.abs(y))),
                        entropy=float(-np.sum(q * np.log(q)) / np.log(len(power))),
                    )
                records.append(r)
    except sqlite3.Error as exc:
        raise ValueError('数据库无法读取或尚未写入完成：' + str(exc)) from exc
    if not records:
        raise ValueError('没有波形记录。')
    full = [r for r in records if r['n'] == 200000]
    if not full:
        return dict(
            name=pri.stem, full_records=0, main=None, alternate_disjoint=None,
            all_full_features={}, peak_band_fraction=0.0, extended=None,
        )
    a = inspect(records)
    extended = None
    main = a['main']
    if main and not main['clipped_windows']:
        missing = [i for i in main['ids'] if i not in samples_by_id]
        if missing:
            raise ValueError('平稳段波形缓存缺失，无法计算扩展特征。')
        extended = _aggregate_extended(main['ids'], samples_by_id)
    # free waveform cache before returning
    samples_by_id.clear()
    return dict(
        name=pri.stem,
        full_records=len(full),
        main=main,
        alternate_disjoint=a['alternate_disjoint'],
        all_full_features=a['all_full_features'],
        peak_band_fraction=float(np.mean([108 <= r['peak_khz'] < 112 for r in full])),
        extended=extended,
    )
