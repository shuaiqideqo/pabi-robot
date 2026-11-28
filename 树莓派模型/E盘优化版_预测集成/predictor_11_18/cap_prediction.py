#!/usr/bin/env python3
"""CAP 结束后的最终预测入口（给 ae_uart_service 调用）。

把 11.18 重拟合 v2 的实验 rawRegressionCm 映射成现有 UART/APP 协议字段：
  { status, result: { distanceCm, nearestReferenceDistanceCm, hasPartialDischarge, distanceText } }

正式 decision 因 deployment_ready=false / 误差带>8 几乎不发布数值；
现场用实验估算 raw 填 cm：0–40 都写数值；近场 st=near，超范围 st=out（>40 仍不填 cm）。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from predict import predict_file  # noqa: E402


REF_DISTANCES = (10, 20, 30, 40)


def _nearest_ref(cm: float | None) -> int | None:
    if cm is None or not math.isfinite(cm):
        return None
    return int(min(REF_DISTANCES, key=lambda d: abs(d - cm)))


def _empty(status: str, text: str = "", has_pd=None) -> dict:
    return {
        "status": status,
        "result": {
            "distanceCm": None,
            "nearestReferenceDistanceCm": None,
            "hasPartialDischarge": has_pd,
            "distanceText": text,
        },
        "modelDataset": "11.18_16V_0-40cm_v2",
        "experimental": True,
    }


def adapt(raw_pred: dict) -> dict:
    """将 predict_file(..., include_experiment=True) 输出适配为旧协议。"""
    status = str(raw_pred.get("status") or "")
    reason = str(raw_pred.get("reason") or "")
    title = str(raw_pred.get("title") or "")
    experiment = raw_pred.get("experiment") or {}
    candidate = experiment.get("candidateDecision") or {}
    cand_status = str(candidate.get("status") or status)
    raw = experiment.get("rawRegressionCm")
    if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
        raw_f = float(raw)
    else:
        raw_f = None

    base = {
        "modelDataset": "11.18_16V_0-40cm_v2",
        "experimental": True,
        "modelId": raw_pred.get("modelId"),
        "v2Status": status,
        "candidateStatus": cand_status,
        "rawRegressionCm": raw_f,
        "stableInterval": raw_pred.get("stableInterval"),
        "reason": reason or candidate.get("reason") or "",
    }

    if status in ("error",) or cand_status == "error":
        out = _empty("prediction_read_error", reason or title or "分析失败", None)
        out.update(base)
        return out

    if cand_status == "unreliable" or status == "unreliable":
        # 信号/平稳段不足 → 对齐旧 nopd/few
        text = reason or candidate.get("reason") or title or "无法可靠判断"
        mapped = "below_noise_or_beyond_50cm" if "信号不足" in text else "insufficient_events"
        out = _empty(mapped, text, False if "信号不足" in text else None)
        out.update(base)
        return out

    # 超范围：仍只报 st=out，不填 cm（数值拟合区为 10–40）
    if cand_status == "outside" or (raw_f is not None and raw_f > 40.0):
        out = _empty(
            "beyond_60cm_outside_display",
            f"疑似超出 40 cm（实验估算 {raw_f:.1f} cm）" if raw_f is not None else "疑似超出 40 cm",
            True,
        )
        out.update(base)
        return out

    if raw_f is None:
        # 近场无 raw 时仍可只报 near
        if cand_status == "near":
            out = _empty("near_10cm_or_near_field", "近场约 0–10 cm", True)
            out.update(base)
            return out
        out = _empty(
            "temporarily_unavailable",
            reason or "无实验估算值",
            None,
        )
        out.update(base)
        return out

    cm = round(raw_f, 1)
    # 0–10：填实验 cm，status 仍为近场 → UART st=near，CSV 有 distanceCm
    if cand_status == "near" or cm < 10.0:
        text = f"{cm:.1f} cm（近场 0–10 实验估算，待独立验证）"
        if reason or candidate.get("reason"):
            text += f"；{reason or candidate.get('reason')}"
        return {
            **base,
            "status": "near_10cm_or_near_field",
            "result": {
                "distanceCm": cm,
                "nearestReferenceDistanceCm": _nearest_ref(cm),
                "hasPartialDischarge": True,
                "distanceText": text,
            },
        }

    # 10–40：实验 raw → cm，st=ok
    text = f"{cm:.1f} cm（11.18 实验估算，待独立验证）"
    if reason or candidate.get("reason"):
        text += f"；{reason or candidate.get('reason')}"
    return {
        **base,
        "status": "within_calibration_range",
        "result": {
            "distanceCm": cm,
            "nearestReferenceDistanceCm": _nearest_ref(cm),
            "hasPartialDischarge": True,
            "distanceText": text,
        },
    }


def predict(pridb_path: str | Path) -> dict:
    path = Path(pridb_path)
    try:
        raw_pred = predict_file(path, include_experiment=True)
    except (ValueError, OSError) as exc:
        return _empty("prediction_read_error", str(exc), None)
    except Exception as exc:  # noqa: BLE001 — CAP 路径不能因未预期异常崩掉服务
        return _empty("prediction_read_error", f"{type(exc).__name__}: {exc}", None)
    return adapt(raw_pred)


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python cap_prediction.py <采集.pridb>", file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(predict(sys.argv[1]), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
