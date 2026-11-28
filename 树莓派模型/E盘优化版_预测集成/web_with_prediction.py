"""Existing device web app plus a read-only 11.18 prediction endpoint.

Run this file in place of web_server.py. The device-control routes and main.py
are imported unchanged from web_server.py.
"""

import csv
import json
import os
import sys
import threading
import time
from pathlib import Path

import psutil
from flask import jsonify, render_template, request

from web_server import DATA_DIR as ORIGINAL_DATA_DIR
from web_server import PID_FILE, app, is_program_running


HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("PREDICTION_DATA_DIR", ORIGINAL_DATA_DIR))
PREDICTOR_DIR = HERE / "predictor_11_18"
PREDICT_PY = PREDICTOR_DIR / "cap_prediction.py"
PRED_LOG_CSV = Path(os.environ.get("AE_PRED_LOG", HERE / "prediction_log.csv"))
PRED_LATEST_JSON = Path(os.environ.get("AE_PRED_LATEST", HERE / "latest_prediction.json"))
_cache_lock = threading.Lock()
_cache_key = None
_cache_result = None
_cache_retry_at = 0.0


def _response(status, message, capturing=False):
    return {
        "phase": "provisional", "status": status, "sourceFile": None,
        "capturing": capturing,
        "eventCount": 0, "waveformCount": 0,
        "result": {"hasPartialDischarge": None, "distanceCm": None, "distanceText": message},
        "quality": {"warnings": []},
        "modelDataset": "11.18_16V_0-40cm_v2",
    }


def _fingerprint(path):
    """Include SQLite WAL files so live writes invalidate the cached result."""
    parts = []
    for candidate in (path, Path(str(path) + "-wal")):
        try:
            stat = candidate.stat()
            parts.append((stat.st_mtime_ns, stat.st_size))
        except FileNotFoundError:
            parts.append(None)
    return tuple(parts)


def _active_capture_start_ns():
    """Only a live main.py process started by the web app counts as capture."""
    if not is_program_running():
        return None
    try:
        pid = int(Path(PID_FILE).read_text(encoding="utf-8").strip())
        process = psutil.Process(pid)
        if process.status() == psutil.STATUS_ZOMBIE:
            return None
        command = process.cmdline()
        if not command or Path(command[-1]).name != "main.py":
            return None
        return int(process.create_time() * 1_000_000_000)
    except (OSError, ValueError, psutil.Error):
        return None


def _latest_pair(after_ns=0):
    if not DATA_DIR.is_dir():
        return None
    files = (p for p in DATA_DIR.glob("*.pridb") if p.is_file())
    dated = ((p, max((x[0] for x in _fingerprint(p) if x), default=0)) for p in files)
    recent = ((p, mtime) for p, mtime in dated if mtime >= after_ns)
    return max(recent, key=lambda item: item[1], default=(None, 0))[0]


def _predict_pridb(pridb, capturing):
    if not PREDICT_PY.is_file():
        result = _response("prediction_unavailable", "找不到 11.18 预测脚本")
    else:
        try:
            if str(PREDICTOR_DIR) not in sys.path:
                sys.path.insert(0, str(PREDICTOR_DIR))
            from cap_prediction import predict as predict_11_18  # type: ignore

            result = predict_11_18(pridb)
        except Exception as exc:  # noqa: BLE001
            result = _response("prediction_unavailable", str(exc)[-300:])
    result["sourceFile"] = pridb.name
    result["capturing"] = capturing
    result["phase"] = "provisional"
    result.setdefault("quality", {"warnings": []})
    return result


@app.route("/api/prediction/file/<path:filename>", methods=["GET"])
def file_prediction(filename):
    """Predict a selected data pair without changing acquisition or saved files."""
    if (not filename or filename in (".", "..") or "/" in filename
            or chr(92) in filename or not filename.endswith(".pridb")):
        return jsonify(_response("invalid_filename", "只能选择数据目录中的 .pridb 文件")), 400
    data_dir = DATA_DIR.resolve()
    pridb = data_dir / filename
    if not pridb.is_file() or pridb.resolve().parent != data_dir:
        return jsonify(_response("file_not_found", "找不到所选 .pridb 文件")), 404
    tradb = pridb.with_suffix(".tradb")
    if not tradb.is_file() or tradb.resolve().parent != data_dir:
        return jsonify(_response("waiting_for_tradb", "等待同名 .tradb 波形文件")), 409
    if not PREDICT_PY.is_file():
        return jsonify(_response("prediction_unavailable", "找不到 11.18 预测脚本")), 503
    return jsonify(_predict_pridb(pridb, is_program_running()))


@app.route("/api/prediction/latest", methods=["GET"])
def latest_prediction():
    global _cache_key, _cache_result, _cache_retry_at

    capture_start_ns = _active_capture_start_ns()
    if capture_start_ns is None:
        return jsonify(_response("not_capturing", "当前未采集；历史文件请在信号文件列表点击「检测结果」"))

    pridb = _latest_pair(after_ns=capture_start_ns)
    if pridb is None:
        return jsonify(_response("waiting_for_new_data", "等待本次采集的新数据", True))
    tradb = pridb.with_suffix(".tradb")
    if not tradb.is_file():
        return jsonify(_response("waiting_for_tradb", "等待同名 .tradb 波形文件", True))
    if not PREDICT_PY.is_file():
        return jsonify(_response("prediction_unavailable", "找不到 11.18 预测脚本", True))

    key = (capture_start_ns, str(pridb), _fingerprint(pridb), _fingerprint(tradb))
    with _cache_lock:
        if key == _cache_key and (
            _cache_result["result"].get("distanceCm") is not None
            or time.monotonic() < _cache_retry_at
        ):
            return jsonify(_cache_result)
        result = _predict_pridb(pridb, True)
        _cache_key, _cache_result = key, result
        _cache_retry_at = time.monotonic() + 5
        return jsonify(result)


def _read_prediction_history(limit: int = 50):
    """Read CAP prediction_log.csv, newest first."""
    if limit < 1:
        limit = 1
    if limit > 500:
        limit = 500
    if not PRED_LOG_CSV.is_file():
        return [], "还没有 prediction_log.csv；等 CAP/KEY5/AUTO 跑完预测后会自动生成。"
    try:
        with PRED_LOG_CSV.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError as exc:
        return [], f"读取失败: {exc}"
    rows.reverse()
    return rows[:limit], None


@app.route("/api/prediction/history", methods=["GET"])
def prediction_history():
    """Historical CAP results from prediction_log.csv for the web panel."""
    try:
        limit = int(request.args.get("limit", "50"))
    except (TypeError, ValueError):
        limit = 50
    rows, err = _read_prediction_history(limit)
    latest = None
    if PRED_LATEST_JSON.is_file():
        try:
            latest = json.loads(PRED_LATEST_JSON.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            latest = None
    return jsonify({
        "ok": err is None,
        "message": err or "",
        "count": len(rows),
        "path": str(PRED_LOG_CSV),
        "latest": latest,
        "rows": rows,
    })


def index_with_prediction():
    return render_template("index_with_prediction.html")


# Keep the existing control routes while changing only the page shown at /.
app.view_functions["index"] = index_with_prediction


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
