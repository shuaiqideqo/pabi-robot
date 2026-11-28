#!/usr/bin/env python3
"""常驻 AE 串口服务: 等子机 CAP → 采集 15s → 预测 → 回传 R 行.

接线(机载直连, 无 USB-TTL):
  子机 PA9(TX)  → Pi Pin10 / GPIO15 (RX)
  子机 PA10(RX) ← Pi Pin8  / GPIO14 (TX)
  共地 + 子机 5V 给 Pi 供电

协议(ASCII, \\n 结尾):
  子机→Pi: CAP
  Pi→子机: R,cm=<f|,>,near=<n|,>,pd=<0|1|,>,st=<code>
  爬行 子机→Pi: P,z=<0.1°>,y=<0.1°>,w=<dps>,a=<轮速>,b=<轮速>,d=<mm>
  爬行 Pi→子机: U,<差速>
"""

from __future__ import annotations

import csv
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
PREDICTOR_DIR = HERE / "predictor_11_18"
PREDICT_PY = PREDICTOR_DIR / "cap_prediction.py"
DATA_DIR = Path(os.environ.get("AE_DATA_DIR", HERE / "data_storage"))
MAIN_PY = HERE / "main.py"
SERIAL_PORT = os.environ.get("AE_UART_PORT", "/dev/serial0")
BAUD = int(os.environ.get("AE_UART_BAUD", "115200"))
CLIMB_LOG_CSV = Path(os.environ.get("AE_CLIMB_LOG", HERE / "climb_pose_log.csv"))
PRED_LOG_CSV = Path(os.environ.get("AE_PRED_LOG", HERE / "prediction_log.csv"))
PRED_LATEST_JSON = Path(os.environ.get("AE_PRED_LATEST", HERE / "latest_prediction.json"))
PRED_LOG_FIELDS = (
    "time",
    "st",
    "status",
    "distanceCm",
    "rawRegressionCm",
    "nearestReferenceDistanceCm",
    "hasPartialDischarge",
    "distanceText",
    "reason",
    "rLine",
    "pridb",
    "modelId",
)

_busy = threading.Lock()
_ser_lock = threading.Lock()
_log_lock = threading.Lock()


def _log(msg: str) -> None:
    print(f"[AE-UART] {msg}", flush=True)


def _open_serial():
    try:
        import serial  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "需要 pyserial: pip3 install pyserial"
        ) from exc
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.2)
    # 丢掉上电噪声
    time.sleep(0.1)
    ser.reset_input_buffer()
    return ser


def _write_line(ser, line: str, quiet: bool = False) -> None:
    payload = (line if line.endswith("\n") else line + "\n").encode("ascii", "ignore")
    with _ser_lock:
        ser.write(payload)
        ser.flush()
    if not quiet:
        _log(f"TX {line.strip()}")


_climb = None
_CLIMB_FIELDS = ("time", "z", "y", "w", "a", "b", "d", "steer")


def _climb_trim():
    global _climb
    if _climb is None:
        from climb_trim import ClimbTrim

        _climb = ClimbTrim()
    return _climb


def _parse_p_line(line: str) -> Optional[dict]:
    if not line.startswith("P,"):
        return None
    out = {}
    for part in line[2:].split(","):
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        try:
            out[key] = float(val)
        except ValueError:
            return None
    if "z" not in out:
        return None
    return out


def _log_climb(row: dict) -> None:
    try:
        with _log_lock:
            new_file = not CLIMB_LOG_CSV.is_file() or CLIMB_LOG_CSV.stat().st_size == 0
            with CLIMB_LOG_CSV.open("a", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_CLIMB_FIELDS, extrasaction="ignore")
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
    except OSError as exc:
        _log(f"写 {CLIMB_LOG_CSV.name} 失败: {exc}")


def _handle_pose(ser, line: str) -> None:
    sample = _parse_p_line(line)
    if sample is None:
        return
    z_deg = sample["z"] / 10.0
    y_deg = sample.get("y", 0.0) / 10.0
    steer = _climb_trim().step(z_deg, time.time())
    sent = int(round(steer))
    _write_line(ser, f"U,{sent}", quiet=True)
    _log_climb(
        {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "z": f"{z_deg:.2f}",
            "y": f"{y_deg:.2f}",
            "w": sample.get("w", ""),
            "a": sample.get("a", ""),
            "b": sample.get("b", ""),
            "d": sample.get("d", ""),
            "steer": sent,
        }
    )


def _latest_pridb(after_mtime: float) -> Optional[Path]:
    if not DATA_DIR.is_dir():
        return None
    best = None
    best_m = after_mtime
    for p in DATA_DIR.glob("*.pridb"):
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if m >= best_m and (best is None or m > best_m):
            best, best_m = p, m
    return best


# 与子机 Ae_StCode / App AE_ST_TEXT / 模型 status 对齐的短码
def _status_code(status: str, has_pd, distance_cm) -> str:
    if status in ("near_source_point", "near_10cm_or_near_field"):
        return "near"  # 近场约0–10cm，无细分数值
    if status in (
        "near_10cm_boundary",
        "within_calibration_range",
        "near_50cm_boundary",
        "beyond_50cm_outside_calibration",
    ):
        return "ok"
    if status in (
        "insufficient_events",
        "collecting",
        "waiting_for_pridb",
        "waiting_for_tradb",
        "no_ae_events",
    ):
        return "few"
    if status in ("incompatible_acquisition",):
        return "bad"
    if status in ("below_noise_or_beyond_50cm",):
        return "nopd"
    if status in ("beyond_60cm_outside_display",):
        return "out"
    if status in ("temporarily_unavailable",):
        return "busy"
    # 有数值时优先 ok，避免 has_pd=False 误盖掉距离
    if distance_cm is not None:
        return "ok"
    if has_pd is False:
        return "nopd"
    return "err"


def _format_result(obj: dict) -> str:
    res = obj.get("result") or {}
    cm = res.get("distanceCm")
    near = res.get("nearestReferenceDistanceCm")
    pd = res.get("hasPartialDischarge")
    st = _status_code(str(obj.get("status") or ""), pd, cm)
    text = res.get("distanceText") or ""
    _log(f"预测 status={obj.get('status')} cm={cm} near={near} pd={pd} → st={st} | {text}")
    cm_s = "" if cm is None else f"{float(cm):.1f}"
    near_s = "" if near is None else str(int(near))
    if pd is True:
        pd_s = "1"
    elif pd is False:
        pd_s = "0"
    else:
        pd_s = ""
    return f"R,cm={cm_s},near={near_s},pd={pd_s},st={st}"


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _write_prediction_files(
    pred: dict,
    *,
    r_line: str,
    pridb: Optional[str] = None,
    st: Optional[str] = None,
) -> None:
    """写 latest_prediction.json，并追加一行到 prediction_log.csv（可读，含成功与失败原因）。"""
    res = pred.get("result") or {}
    cm = res.get("distanceCm")
    pd = res.get("hasPartialDischarge")
    status = str(pred.get("status") or "")
    if st is None:
        st = _status_code(status, pd, cm)
    now = time.time()
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    reason = (
        pred.get("reason")
        or res.get("distanceText")
        or pred.get("error")
        or ""
    )
    row = {
        "time": local_time,
        "st": st,
        "status": status,
        "distanceCm": _cell(cm),
        "rawRegressionCm": _cell(pred.get("rawRegressionCm")),
        "nearestReferenceDistanceCm": _cell(res.get("nearestReferenceDistanceCm")),
        "hasPartialDischarge": _cell(pd),
        "distanceText": res.get("distanceText") or "",
        "reason": reason,
        "rLine": r_line.strip(),
        "pridb": pridb or "",
        "modelId": pred.get("modelId") or pred.get("modelDataset") or "11.18",
    }
    latest = {
        **row,
        "updated_at": now,
        "experimental": bool(pred.get("experimental")),
        "stableInterval": pred.get("stableInterval"),
        "candidateStatus": pred.get("candidateStatus"),
        "v2Status": pred.get("v2Status"),
    }
    try:
        PRED_LATEST_JSON.write_text(
            json.dumps(latest, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    except OSError as exc:
        _log(f"写 {PRED_LATEST_JSON.name} 失败: {exc}")
    try:
        with _log_lock:
            new_file = not PRED_LOG_CSV.is_file() or PRED_LOG_CSV.stat().st_size == 0
            with PRED_LOG_CSV.open("a", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=PRED_LOG_FIELDS, extrasaction="ignore")
                if new_file:
                    writer.writeheader()
                writer.writerow({k: row.get(k, "") for k in PRED_LOG_FIELDS})
        _log(
            f"预测已写入 {PRED_LOG_CSV.name} @ {local_time} "
            f"st={st} cm={row['distanceCm'] or '-'} raw={row['rawRegressionCm'] or '-'}"
        )
    except OSError as exc:
        _log(f"写 {PRED_LOG_CSV.name} 失败: {exc}")


def _synthetic_pred(status: str, text: str, st: str) -> dict:
    return {
        "status": status,
        "reason": text,
        "result": {
            "distanceCm": None,
            "nearestReferenceDistanceCm": None,
            "hasPartialDischarge": None,
            "distanceText": text,
        },
        "modelId": "11.18",
        "_forced_st": st,
    }


def _run_capture() -> bool:
    if not MAIN_PY.is_file():
        _log(f"找不到 {MAIN_PY}")
        return False
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["AE_DATA_DIR"] = str(DATA_DIR)
    _log(f"开始采集 main.py … data={DATA_DIR}")
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-u", str(MAIN_PY)],
        cwd=str(HERE),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    if proc.stdout:
        for line in proc.stdout.strip().splitlines()[-40:]:
            _log(f"main| {line}")
    if proc.stderr:
        for line in proc.stderr.strip().splitlines()[-20:]:
            _log(f"main! {line}")
    _log(f"采集结束 code={proc.returncode}")
    try:
        files = sorted(DATA_DIR.glob("AE_Data_*.pridb"), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            newest = files[0]
            _log(f"最新 pridb={newest.name} size={newest.stat().st_size}")
        else:
            _log("data_storage 内仍无 AE_Data_*.pridb")
    except OSError as exc:
        _log(f"列举数据目录失败: {exc}")
    try:
        os.sync()
    except Exception:
        pass
    return proc.returncode == 0


def _run_predict(pridb: Path) -> dict:
    empty = {
        "status": "prediction_unavailable",
        "result": {"distanceCm": None, "nearestReferenceDistanceCm": None, "hasPartialDischarge": None},
    }
    if not PREDICT_PY.is_file():
        return empty
    try:
        if str(PREDICTOR_DIR) not in sys.path:
            sys.path.insert(0, str(PREDICTOR_DIR))
        from cap_prediction import predict as predict_11_18  # type: ignore

        return predict_11_18(pridb)
    except Exception as exc:  # noqa: BLE001
        _log(f"预测失败(11.18): {exc}")
        return {
            "status": "prediction_read_error",
            "result": {"distanceCm": None, "nearestReferenceDistanceCm": None, "hasPartialDischarge": None},
        }


def _handle_cap(ser) -> None:
    if not _busy.acquire(blocking=False):
        line = "R,cm=,near=,pd=,st=busy"
        _write_line(ser, line)
        _write_prediction_files(
            _synthetic_pred("temporarily_unavailable", "服务忙，上一轮 CAP 未结束", "busy"),
            r_line=line,
            st="busy",
        )
        return
    try:
        _write_line(ser, "OK CAP")
        t0 = time.time()
        ok = _run_capture()
        if not ok:
            line = "R,cm=,near=,pd=,st=err"
            _write_line(ser, line)
            _write_prediction_files(
                _synthetic_pred("capture_failed", "main.py 采集失败", "err"),
                r_line=line,
                st="err",
            )
            return
        pridb = _latest_pridb(t0 - 1.0)
        if pridb is None:
            line = "R,cm=,near=,pd=,st=few"
            _write_line(ser, line)
            _write_prediction_files(
                _synthetic_pred("waiting_for_pridb", "采集后未找到新的 .pridb", "few"),
                r_line=line,
                st="few",
            )
            return
        # 等 tradb 落盘片刻
        tradb = pridb.with_suffix(".tradb")
        for _ in range(20):
            if tradb.is_file() and tradb.stat().st_size > 0:
                break
            time.sleep(0.1)
        pred = _run_predict(pridb)
        line = _format_result(pred)
        _write_prediction_files(pred, r_line=line, pridb=pridb.name)
        # 多发两次, 提高子机在抬升/视觉阶段抽到的概率
        _write_line(ser, line)
        time.sleep(0.05)
        _write_line(ser, line)
    finally:
        _busy.release()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _log(f"常驻启动 port={SERIAL_PORT} baud={BAUD}")
    _log(f"predictor=11.18 py={PREDICT_PY}")
    _log(f"pred_log={PRED_LOG_CSV} latest={PRED_LATEST_JSON}")
    ser = _open_serial()
    buf = ""
    while True:
        try:
            chunk = ser.read(64)
            if chunk:
                buf += chunk.decode("ascii", "ignore")
                while True:
                    nl = buf.find("\n")
                    if nl < 0:
                        # 也接受裸 \r
                        cr = buf.find("\r")
                        if cr < 0:
                            break
                        nl = cr
                    line = buf[:nl].strip()
                    buf = buf[nl + 1 :]
                    if not line:
                        continue
                    if line.startswith("P,"):
                        _handle_pose(ser, line)
                        continue
                    _log(f"RX {line}")
                    if line.upper() == "CAP":
                        threading.Thread(target=_handle_cap, args=(ser,), daemon=True).start()
                    elif line.upper() in ("PING", "?"):
                        _write_line(ser, "OK PING")
            else:
                time.sleep(0.02)
        except KeyboardInterrupt:
            _log("退出")
            break
        except Exception as exc:
            _log(f"串口异常: {exc}")
            time.sleep(1.0)
            try:
                ser.close()
            except Exception:
                pass
            try:
                ser = _open_serial()
                buf = ""
            except Exception as exc2:
                _log(f"重开串口失败: {exc2}")
                time.sleep(2.0)


if __name__ == "__main__":
    main()
