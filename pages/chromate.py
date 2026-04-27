import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
import shutil
import tempfile
from typing import Optional
from tensorflow.keras.models import load_model
from zoneinfo import ZoneInfo
import h5py
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

from datetime import datetime
import time
from openai_service2 import generate_anomaly_report

# =========================================================
# 파일 경로 설정
# =========================================================
MODEL_PATH = "hybrid_model.h5"
ARTIFACT_PATH = "hybrid_artifacts.pkl"
RAW_DATA_PATH = "kemp-abh-sensor-final1.csv"

SERIES_BASE_COLOR = "#4A4A4A"
ALL_LOT_SERIES_OPACITY = 0.85
SINGLE_LOT_SERIES_OPACITY = 1.0

# =========================================================
# 1. 페이지 설정 및 스타일
# =========================================================
st.set_page_config(layout="wide")

st.set_page_config(layout="wide")
OPERATORS = ["손준호", "최윤호", "손영민", "허가은", "신소망", "윤다빈"]
NOTIFY_TARGETS = ["김윤환", "손보미", "신재춘", "박영선", "김준성"]

if "logs" not in st.session_state:
    st.session_state.logs = []

if "ai_draft" not in st.session_state:
    st.session_state.ai_draft = ""


header_left, header_center, header_right = st.columns([1.2, 8, 1.2])

with header_left:
    if st.button(
        "← 메인",
        key="top_main_btn",
        use_container_width=True,
        type="primary"
    ):
        st.switch_page("dashboard.py")

with header_right:
    if st.button(
        "불량 탐지 →",
        key="top_detect_btn",
        use_container_width=True,
        type="primary"
    ):
        st.switch_page("pages/app_lot_integrated.py")

with header_center:
    st.markdown(
        """
        <h1 style='text-align: center; margin: 0; white-space: nowrap;'>
            크로메이트 공정 실시간 운영 모니터링
        </h1>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
    .stApp { background-color: #F0EEF3 !important; }

    div[data-testid="stMetric"] {
        background-color: white !important;
        border-radius: 10px !important;
        padding: 15px 20px !important;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.05) !important;
        border: 1px solid #e0e0e0 !important;
    }

    div[data-baseweb="input"],
    div[data-baseweb="select"] > div {
        background-color: white !important;
        border: 1px solid #cccccc !important;
        border-radius: 8px !important;
        min-height: 42px !important;
        box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.03) !important;
        box-sizing: border-box !important;
    }
    div[data-baseweb="base-input"] { background-color: white !important; }
    .stTextInput input {
        background-color: white !important;
        color: black !important;
        border: none !important;
        padding: 0 15px !important;
    }
    div[data-baseweb="select"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    input[disabled] {
        -webkit-text-fill-color: black !important;
        color: black !important;
        background-color: white !important;
        opacity: 1 !important;
    }

    div[data-testid="stTable"] > table {
        background-color: white !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.03) !important;
        border-collapse: collapse !important;
        border: 1px solid #e0e0e0 !important;
    }
    div[data-testid="stTable"] th {
        background-color: #f8f9fa !important;
        color: black !important;
        font-weight: bold !important;
        border-bottom: 2px solid #ddd !important;
    }
    div[data-testid="stTable"] td {
        background-color: white !important;
        color: black !important;
        border-bottom: 1px solid #eee !important;
    }

    div[data-testid="column"] .stButton > button[key^="grid_lot_"] {
        background-color: white !important;
        border: 1px solid #cccccc !important;
        border-radius: 8px !important;
        height: 42px !important;
        font-size: 14px !important;
        font-weight: bold !important;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        box-shadow: 0px 2px 4px rgba(0,0,0,0.05) !important;
    }

    div[data-testid="column"] .stButton > button[key^="grid_lot_"]:hover {
        border-color: #333 !important;
        background-color: #f9f9f9 !important;
    }

    .stButton > button {
        width: 100%;
        height: 50px;
        color: black !important;
        background-color: white !important;
        border: 1px solid #ccc !important;
        font-weight: bold;
        border-radius: 8px !important;
    }



    .process-stepper-wrap {
        margin: 8px 0 18px 0;
        padding: 0;
        background: transparent;
        border: none;
        box-shadow: none;
    }

    .process-flow {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        width: 100%;
    }

    .process-step {
        flex: 0 0 96px;
        width: 110px;
        height: 50px;
        background: white;
        border: 1px solid #d7dfe9;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0px 2px 6px rgba(60, 70, 90, 0.05);
    }

    .process-step.active {
        background: #edf4ff;
        border: 2px solid #4d7df3;
        box-shadow: 0px 0px 0px 3px rgba(77, 125, 243, 0.14);
    }

    .process-step-name {
        font-size: 17px;
        font-weight: 800;
        color: #222 !important;
        white-space: nowrap;
    }

    .process-step.active .process-step-name {
        color: #356eea !important;
    }

    .process-arrow {
        flex: 0 0 auto;
        font-size: 22px;
        font-weight: 800;
        color: #9aa8bb !important;
        line-height: 1;
    }

    @media (max-width: 1100px) {
        .process-flow { gap: 5px; }
        .process-step { flex-basis: 82px; width: 82px; height: 34px; }
        .process-step-name { font-size: 12px; }
        .process-arrow { font-size: 18px; }
    }


    h1, h2, h3, p, span, label { color: black !important; }
    hr { border-top: 1px solid #d5d1db !important; margin: 20px 0 !important; }

    /* 상단 이동 버튼만 적용 */
    .stButton > button[kind="primary"] {
        background-color: #F0EEF3 !important;
        color: #212529 !important;
        border: 1px solid #d5d1db !important;
        font-weight: bold !important;
    }

    .stButton > button[kind="primary"] * {
        color: #212529 !important;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #e6e1ec !important;
        border-color: #c9c1d1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 2. 유틸 함수
# =========================================================
def is_action_completed(logs, process_name, lot, status, selected_date):
    if not logs:
        return False

    selected_date_str = str(selected_date)

    for log in logs:
        log_time = log.get("일시", "")

        try:
            log_date_str = str(pd.to_datetime(log_time).date())
        except Exception:
            log_date_str = ""

        if (
            str(log.get("공정", "")) == str(process_name)
            and str(log.get("Lot", "")) == str(lot)
            and str(log.get("상태", "")) == str(status)
            and log_date_str == selected_date_str
        ):
            return True

    return False

def zscore_with_ref(ref_score, target_score):
    ref_score = np.asarray(ref_score, dtype=np.float32)
    target_score = np.asarray(target_score, dtype=np.float32)
    mean_ = np.mean(ref_score)
    std_ = np.std(ref_score) + 1e-8
    return (target_score - mean_) / std_


def determine_lot_col(df: pd.DataFrame) -> str:
    if "LOT_ID" in df.columns:
        return "LOT_ID"
    if "Lot" in df.columns:
        return "Lot"
    raise ValueError("원본 데이터에 LOT_ID 또는 Lot 컬럼이 필요합니다.")


def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "Datetime" not in df.columns:
        raise ValueError("원본 데이터에 Datetime 컬럼이 필요합니다.")
    out = df.copy()
    out["Datetime"] = pd.to_datetime(out["Datetime"], errors="coerce")
    out = out.dropna(subset=["Datetime"]).copy()
    out["DATE_ONLY"] = out["Datetime"].dt.date
    return out


def infer_feature_column(df: pd.DataFrame, feature_name: str, group_col: str) -> pd.Series:
    if feature_name in df.columns:
        return df[feature_name]
    if feature_name.endswith("_abs_diff"):
        base = feature_name[:-9]
        if base in df.columns:
            return df.groupby(group_col)[base].diff().abs().fillna(0)
    if feature_name.endswith("_acc"):
        base = feature_name[:-4]
        if base in df.columns:
            return df.groupby(group_col)[base].diff().diff().fillna(0)
    if "_roll_std_" in feature_name:
        base, win = feature_name.split("_roll_std_")
        if base in df.columns and win.isdigit():
            return df.groupby(group_col)[base].transform(
                lambda s: s.rolling(int(win), min_periods=1).std()
            ).fillna(0)
    if "_roll_mean_" in feature_name:
        base, win = feature_name.split("_roll_mean_")
        if base in df.columns and win.isdigit():
            return df.groupby(group_col)[base].transform(
                lambda s: s.rolling(int(win), min_periods=1).mean()
            ).bfill().ffill()
    raise ValueError(f"필수 피처 '{feature_name}' 를 만들 수 없습니다.")


def build_requested_features(df: pd.DataFrame, feature_list: list[str], group_col: str) -> pd.DataFrame:
    out = df.copy()
    for feat in feature_list:
        if feat not in out.columns:
            out[feat] = infer_feature_column(out, feat, group_col)
    return out


def make_sequences(arr: np.ndarray, window_size: int):
    x_list, y_list, row_idx = [], [], []
    for i in range(window_size, len(arr)):
        x_list.append(arr[i - window_size: i])
        y_list.append(arr[i])
        row_idx.append(i)
    return np.asarray(x_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32), row_idx


def score_to_votes(score: float, threshold: float) -> int:
    margin = float(score - threshold)
    if margin < 0:
        return 4
    if margin < 0.5:
        return 3
    if margin < 1.0:
        return 2
    if margin < 1.5:
        return 1
    return 0


def add_new_alert_flags(df: pd.DataFrame, group_col: str, patience: int = 3) -> pd.DataFrame:
    out = df.sort_values([group_col, "Datetime"]).copy()
    is_anomaly = (out["final_label"] == "ANOMALY").astype(int)
    roll_sum = is_anomaly.groupby(out[group_col]).rolling(window=patience, min_periods=1).sum()
    roll_sum = roll_sum.reset_index(level=0, drop=True)
    out["is_confirmed_anomaly"] = (roll_sum >= patience)
    out["prev_confirmed"] = out.groupby(group_col)["is_confirmed_anomaly"].shift(1).fillna(False)
    out["is_new_alert"] = out["is_confirmed_anomaly"] & (~out["prev_confirmed"])
    out["alert_event_id"] = out.groupby(group_col)["is_new_alert"].cumsum() # 이벤트 고유 ID 부여
    out.drop(columns=["prev_confirmed"], inplace=True)
    return out


def add_daily_alert_flags(df: pd.DataFrame, group_col: str, patience: int = 3) -> pd.DataFrame:
    out = df.sort_values([group_col, "DATE_ONLY", "Datetime"]).copy()
    is_anomaly = (out["final_label"] == "ANOMALY").astype(int)
    roll_sum = is_anomaly.groupby(out[group_col]).rolling(window=patience, min_periods=1).sum()
    roll_sum = roll_sum.reset_index(level=0, drop=True)
    out["is_confirmed_anomaly"] = (roll_sum >= patience)
    out["prev_confirmed_same_day"] = out.groupby([group_col, "DATE_ONLY"])["is_confirmed_anomaly"].shift(1).fillna(False)
    out["is_daily_new_alert"] = out["is_confirmed_anomaly"] & (~out["prev_confirmed_same_day"])
    out["alert_event_id"] = out.groupby(group_col)["is_daily_new_alert"].cumsum() # 이벤트 고유 ID 부여
    out.drop(columns=["prev_confirmed_same_day"], inplace=True)
    return out


def compute_risk_index(score_series: pd.Series, threshold: float) -> pd.Series:
    if score_series.empty:
        return pd.Series(dtype=float)
    upper = float(score_series.quantile(0.95))
    if upper <= threshold:
        upper = threshold + 1.0
    risk = (score_series.astype(float) - float(threshold)) / max(upper - float(threshold), 1e-6) * 100.0
    return risk.clip(lower=0, upper=100)


def risk_color(score: float, threshold: float, severe_margin: float = 1.0) -> str:
    if score < threshold:
        return "#00cc66"
    if score < threshold + severe_margin:
        return "#ffcc00"
    return "#ff4b4b"


def get_status_label(score: float, threshold: float, severe_margin: float = 1.0) -> str:
    if score < threshold:
        return "정상"
    if score < threshold + severe_margin:
        return "주의"
    return "위험"


def get_status_badge(score: float, threshold: float, severe_margin: float = 1.0) -> str:
    label = get_status_label(score, threshold, severe_margin=severe_margin)
    if label == "정상":
        return "✅ 정상"
    if label == "주의":
        return "⚠️ 주의"
    return "🚨 위험"


def process_state_color(state: str) -> str:
    mapping = {
        "대기": "#b7b7b7",
        "진행중": "#5bc0de",
        "완료": "#7f8c8d",
    }
    return mapping.get(state, "#b7b7b7")


def get_lot_signal_color(process_state: str, risk_status: str, latest_score, threshold: float) -> str:
    if process_state == "대기":
        return "#c8c8c8"
    if process_state == "진행중":
        return "#5bc0de"

    if risk_status == "정상":
        return "#00cc66"
    elif risk_status == "주의":
        return "#ffcc00"
    elif risk_status == "이상":
        return "#ff4b4b"

    if pd.isna(latest_score):
        return "#c8c8c8"
    return risk_color(float(latest_score), threshold)


def get_lot_signal_text(process_state: str, display_score, threshold: float, planned_start, raw_count: int, window_size: int) -> str:
    if process_state == "대기":
        return planned_start.strftime("예정 %H:%M") if pd.notna(planned_start) else "대기"
    if process_state == "진행중":
        return "진행중"
    if pd.isna(display_score):
        return "-"
    return f"Max {float(display_score):.2f}"


def format_timedelta_hhmmss(delta: pd.Timedelta) -> str:
    total_seconds = max(int(delta.total_seconds()), 0)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def estimate_ongoing_anomaly_duration(target_df: pd.DataFrame) -> str:
    if target_df.empty:
        return "00:00:00"
    temp = target_df.sort_values("Datetime").reset_index(drop=True)
    if temp["final_label"].iloc[-1] != "ANOMALY":
        return "00:00:00"
    start_idx = len(temp) - 1
    while start_idx > 0 and temp["final_label"].iloc[start_idx - 1] == "ANOMALY":
        start_idx -= 1
    delta = temp["Datetime"].iloc[-1] - temp["Datetime"].iloc[start_idx]
    return format_timedelta_hhmmss(delta)


def build_rule_based_signals(row: pd.Series) -> list[str]:
    signals = []
    if "Voltage" in row.index and pd.notna(row["Voltage"]) and row["Voltage"] > 19:
        signals.append("전압 기준 초과")
    if "pH" in row.index and pd.notna(row["pH"]) and (row["pH"] < 1.9 or row["pH"] > 2.2):
        signals.append("pH 기준 이탈")
    if "Temp" in row.index and pd.notna(row["Temp"]) and row["Temp"] > 44:
        signals.append("온도 기준 초과")
    return signals


def zscore_with_ref_single(ref_score, target_score):
    ref_score = np.asarray(ref_score, dtype=np.float32)
    target_score = np.asarray(target_score, dtype=np.float32)
    mean_ = np.mean(ref_score)
    std_ = np.std(ref_score) + 1e-8
    return (target_score - mean_) / std_


def get_single_window_scores(
    model,
    x_window: np.ndarray,
    y_next_true: np.ndarray,
    train_ae_score_ref,
    train_fc_score_ref,
    ae_weight: float = 0.5,
    fc_weight: float = 0.5,
):
    x_input = np.expand_dims(x_window, axis=0)
    recon_pred, forecast_pred = model.predict(x_input, verbose=0)

    ae_score = float(np.mean((recon_pred[0, -1, :] - x_window[-1, :]) ** 2))
    fc_score = float(np.mean((forecast_pred[0] - y_next_true) ** 2))

    ae_z = float(zscore_with_ref_single(train_ae_score_ref, np.array([ae_score]))[0])
    fc_z = float(zscore_with_ref_single(train_fc_score_ref, np.array([fc_score]))[0])
    combined_score = float(ae_weight) * ae_z + float(fc_weight) * fc_z

    return {
        "recon_pred": recon_pred[0],
        "forecast_pred": forecast_pred[0],
        "ae_score": ae_score,
        "fc_score": fc_score,
        "ae_z": ae_z,
        "fc_z": fc_z,
        "combined_score": combined_score,
    }


def occlusion_importance_by_feature(
    model,
    x_window: np.ndarray,
    y_next_true: np.ndarray,
    feature_names: list[str],
    baseline_vector: np.ndarray,
    train_ae_score_ref,
    train_fc_score_ref,
    ae_weight: float = 0.5,
    fc_weight: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    original = get_single_window_scores(
        model=model,
        x_window=x_window,
        y_next_true=y_next_true,
        train_ae_score_ref=train_ae_score_ref,
        train_fc_score_ref=train_fc_score_ref,
        ae_weight=ae_weight,
        fc_weight=fc_weight,
    )

    results = []
    for j, feat in enumerate(feature_names):
        x_masked = x_window.copy()
        x_masked[:, j] = baseline_vector[j]

        masked = get_single_window_scores(
            model=model,
            x_window=x_masked,
            y_next_true=y_next_true,
            train_ae_score_ref=train_ae_score_ref,
            train_fc_score_ref=train_fc_score_ref,
            ae_weight=ae_weight,
            fc_weight=fc_weight,
        )

        results.append(
            {
                "feature": feat,
                "importance_combined": original["combined_score"] - masked["combined_score"],
                "importance_ae": original["ae_score"] - masked["ae_score"],
                "importance_fc": original["fc_score"] - masked["fc_score"],
                "original_combined_score": original["combined_score"],
                "masked_combined_score": masked["combined_score"],
            }
        )

    result_df = (
        pd.DataFrame(results)
        .sort_values("importance_combined", ascending=False)
        .reset_index(drop=True)
    )
    return result_df, original


def map_feature_to_sensor(feature_name: str) -> str:
    if feature_name.startswith("pH"):
        return "pH"
    if feature_name.startswith("Temp"):
        return "Temp"
    if feature_name.startswith("Voltage"):
        return "Voltage"
    return feature_name


def sensor_display_name(sensor_name: str) -> str:
    mapping = {
        "pH": "pH",
        "Temp": "온도",
        "Voltage": "전압",
    }
    return mapping.get(sensor_name, sensor_name)


def feature_reason_phrase(feature_name: str) -> str:
    sensor_name = sensor_display_name(map_feature_to_sensor(feature_name))
    if "_abs_diff" in feature_name:
        return f"{sensor_name} 순간 변화량 급증"
    if "_acc" in feature_name:
        return f"{sensor_name} 변화 흐름 불안정"
    if "_roll_std_" in feature_name:
        return f"{sensor_name} 단기 변동성 증가"
    if "_roll_mean_" in feature_name:
        return f"{sensor_name} 단기 평균 수준 변화"
    return f"{sensor_name} 수준 자체 이상"


def find_representative_anomaly_row(
    score_lot_df: pd.DataFrame, event_time: pd.Timestamp
) -> Optional[pd.Series]:
    if score_lot_df.empty:
        return None

    temp = score_lot_df.sort_values("Datetime").reset_index(drop=True).copy()
    match_idx = temp.index[temp["Datetime"] == pd.Timestamp(event_time)]
    if len(match_idx) == 0:
        return None

    pos = int(match_idx[0])

    start = pos
    while start > 0 and temp.loc[start - 1, "final_label"] == "ANOMALY":
        start -= 1

    end = pos
    while end + 1 < len(temp) and temp.loc[end + 1, "final_label"] == "ANOMALY":
        end += 1

    segment_df = temp.iloc[start:end + 1].copy()
    if segment_df.empty:
        return temp.loc[pos]

    rep_idx = segment_df["combined_score"].idxmax()
    return segment_df.loc[rep_idx]


def build_event_root_cause_info(
    event_row: pd.Series,
    raw_df: pd.DataFrame,
    snapshot_infer_df: pd.DataFrame,
    model,
    meta: dict,
    top_n: int = 3,
) -> dict:
    rule_signals = build_rule_based_signals(event_row)

    def rule_pattern(default_text: str = "패턴 미검출") -> str:
        return ", ".join(rule_signals[:2]) if rule_signals else default_text

    required_meta = ["FEATURES", "WINDOW_SIZE", "train_ae_score", "train_fc_score", "AE_SCORE_WEIGHT", "FC_SCORE_WEIGHT"]

    if any(k not in meta for k in required_meta):
        return {
            "주요 원인 변수": "분석 정보 부족",
            "주요 원인 패턴": rule_pattern("패턴 미검출"),
        }

    lot = str(event_row["Lot"])
    event_time = pd.Timestamp(event_row["Datetime"])
    target_date = event_time.date()

    score_lot_df = snapshot_infer_df[
        (snapshot_infer_df["Lot"].astype(str) == lot) &
        (snapshot_infer_df["Datetime"].dt.date == target_date)
    ].copy()

    representative_row = find_representative_anomaly_row(score_lot_df, event_time)
    target_time = event_time if representative_row is None else pd.Timestamp(representative_row["Datetime"])

    feature_names = list(meta["FEATURES"])
    window_size = int(meta["WINDOW_SIZE"])

    lot_raw_df = raw_df[
        (raw_df["Lot"].astype(str) == lot) &
        (raw_df["Datetime"].dt.date == target_date)
    ].copy().sort_values("Datetime").reset_index(drop=True)

    if lot_raw_df.empty:
        return {
            "주요 원인 변수": "원본 데이터 없음",
            "주요 원인 패턴": rule_pattern("패턴 미검출"),
        }

    if len(lot_raw_df) <= window_size:
        return {
            "주요 원인 변수": "분석 구간 부족",
            "주요 원인 패턴": rule_pattern("패턴 미검출"),
        }

    lot_raw_df = build_requested_features(lot_raw_df, feature_names, group_col="Lot")

    # 여기 중요: warning 줄이려고 DataFrame 형태로 scaler.transform
    scaler = meta.get("scaler") or meta.get("SCALER")
    if scaler is not None:
        feature_df = lot_raw_df[feature_names].copy()
        feature_arr = scaler.transform(feature_df).astype(np.float32)
        baseline_vector = np.zeros(feature_arr.shape[1], dtype=np.float32)
    else:
        feature_arr = lot_raw_df[feature_names].to_numpy(dtype=np.float32)
        baseline_vector = np.nanmean(feature_arr, axis=0).astype(np.float32)

    match_idx = lot_raw_df.index[lot_raw_df["Datetime"] == target_time]
    if len(match_idx) == 0:
        return {
            "주요 원인 변수": "시점 매칭 실패",
            "주요 원인 패턴": rule_pattern("패턴 미검출"),
        }

    target_pos = int(match_idx[0])
    if target_pos < window_size:
        return {
            "주요 원인 변수": "초기 구간 분석 제한",
            "주요 원인 패턴": rule_pattern("패턴 미검출"),
        }

    x_window = feature_arr[target_pos - window_size:target_pos]
    y_next_true = feature_arr[target_pos]

    importance_df, original = occlusion_importance_by_feature(
        model=model,
        x_window=x_window,
        y_next_true=y_next_true,
        feature_names=feature_names,
        baseline_vector=baseline_vector,
        train_ae_score_ref=meta["train_ae_score"],
        train_fc_score_ref=meta["train_fc_score"],
        ae_weight=float(meta["AE_SCORE_WEIGHT"]),
        fc_weight=float(meta["FC_SCORE_WEIGHT"]),
    )

    positive_df = importance_df[importance_df["importance_combined"] > 0].copy()
    if positive_df.empty:
        return {
            "주요 원인 변수": "원인 변수 미확정",
            "주요 원인 패턴": rule_pattern("기여 패턴 미검출"),
        }

    positive_df["sensor_group"] = positive_df["feature"].apply(map_feature_to_sensor)
    sensor_rank = (
        positive_df.groupby("sensor_group")["importance_combined"]
        .sum()
        .sort_values(ascending=False)
    )

    top_sensor = sensor_display_name(sensor_rank.index[0]) if len(sensor_rank) > 0 else "원인 변수 미확정"

    top_phrases = positive_df["feature"].head(top_n).apply(feature_reason_phrase).tolist()

    unique_phrases = []
    for phrase in top_phrases:
        if phrase not in unique_phrases:
            unique_phrases.append(phrase)

    pattern_text = ", ".join(unique_phrases[:2]) if unique_phrases else rule_pattern("기여 패턴 미검출")

    return {
        "주요 원인 변수": top_sensor,
        "주요 원인 패턴": pattern_text,
    }


def safe_sensor_columns(df: pd.DataFrame) -> list[str]:
    preferred = ["pH", "Temp", "Voltage"]
    return [c for c in preferred if c in df.columns]


def natural_lot_key(value):
    s = str(value)
    if s.isdigit():
        return (0, int(s), s)
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        return (1, int(digits), s)
    return (2, s)


def format_datetime_label(dt_value: pd.Timestamp) -> str:
    return pd.Timestamp(dt_value).strftime("%H:%M:%S")


def format_score_value(value) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.2f}"


def sync_selected_lot_from_filter():
    st.session_state["selected_lot"] = st.session_state["selected_lot_chart_filter"]


def build_index_snapshot(full_day_raw: pd.DataFrame, current_idx: int):
    if full_day_raw.empty:
        return full_day_raw.copy(), pd.NaT, "-", "대기"

    safe_idx = max(1, min(int(current_idx), len(full_day_raw)))
    snapshot_raw = full_day_raw.iloc[:safe_idx].copy()
    selected_snapshot = pd.Timestamp(snapshot_raw["Datetime"].max())
    snapshot_label = selected_snapshot.strftime("%H:%M:%S")
    dashboard_mode = "완료" if safe_idx >= len(full_day_raw) else "진행중"
    return snapshot_raw, selected_snapshot, snapshot_label, dashboard_mode

def build_planned_lot_list(full_day_raw: pd.DataFrame) -> list[str]:
    lots = sorted(full_day_raw["Lot"].astype(str).unique().tolist(), key=natural_lot_key)
    return lots


def determine_process_state(raw_snap_lot: pd.DataFrame, raw_full_lot: pd.DataFrame, infer_snap_lot: pd.DataFrame) -> str:
    if raw_snap_lot.empty:
        return "대기"
    if len(raw_snap_lot) < len(raw_full_lot):
        return "진행중"
    return "완료"


def get_lot_risk_status(max_score: float, alert_event_count: int, severe_alert_count: int = 0) -> str:
    alert_event_count = int(alert_event_count)
    severe_alert_count = int(severe_alert_count)
    max_score = float(max_score)

    # severe가 실제로 잡힌 경우만 이상
    if severe_alert_count >= 2:
        return "이상"

    # 일반 threshold도 안 넘으면 정상
    if max_score < 3.79:
        return "정상"

    # 반복 alert가 있으면 주의
    if alert_event_count >= 2:
        return "주의"

    return "정상"

def build_snapshot_lot_stats(snapshot_infer: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if snapshot_infer.empty:
        return pd.DataFrame(columns=["LatestScore", "RiskIndex", "Status", "MaxScore", "SevereAlertCount"])
        
    # 각 로트별로 "구간(이벤트) 최고 점수가 5.7 이상인 이벤트 수" 계산
    def calc_severe_count(g, patience: int = 3):
        temp = g.sort_values("Datetime").copy()
        temp["is_anom"] = (temp["final_label"] == "ANOMALY").astype(int)

        if temp["is_anom"].sum() == 0:
            return 0

        temp["segment_start"] = (
            (temp["is_anom"] == 1) &
            (temp["is_anom"].shift(1, fill_value=0) == 0)
        ).astype(int)
        temp["anom_segment_id"] = temp["segment_start"].cumsum()

        anom = temp[temp["is_anom"] == 1].copy()
        if anom.empty:
            return 0

        seg_stats = anom.groupby("anom_segment_id").agg(
            seg_len=("combined_score", "size"),
            seg_max=("combined_score", "max")
        )

        return int(((seg_stats["seg_len"] >= patience) & (seg_stats["seg_max"] >= 5.7)).sum())

    severe_counts = snapshot_infer.groupby("Lot").apply(calc_severe_count).reset_index(name="SevereAlertCount")

    latest_stats = (
        snapshot_infer.sort_values("Datetime")
        .groupby("Lot")
        .agg(
            LatestScore=("combined_score", "last"),
            MaxScore=("combined_score", "max"),
            AlertCount=("is_daily_new_alert", "sum"),
            AnyAnomaly=("final_label", lambda s: bool((s == "ANOMALY").any())),
        )
    )
    
    latest_stats = latest_stats.merge(severe_counts.set_index("Lot"), left_index=True, right_index=True, how="left")
    latest_stats["SevereAlertCount"] = latest_stats["SevereAlertCount"].fillna(0)
    
    latest_stats["RiskIndex"] = compute_risk_index(latest_stats["LatestScore"], threshold)
    latest_stats["Status"] = latest_stats.apply(
        lambda row: get_lot_risk_status(row["MaxScore"], row["AlertCount"], row["SevereAlertCount"]), axis=1
    )
    return latest_stats.sort_values(by=["LatestScore", "AlertCount"], ascending=False)


def build_lot_risk_stats(
    snapshot_infer: pd.DataFrame,
    board_df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    columns = [
        "Lot", "LatestScore", "MaxScore", "AlertEventCount", "SevereAlertCount", "AnomalyRowCount",
        "TotalScoredRows", "AnomalyRatio", "LotRiskScore", "Status", "LastAlertTime"
    ]
    if snapshot_infer.empty or board_df.empty:
        return pd.DataFrame(columns=columns)

    completed_lots = board_df.loc[board_df["process_state"] == "완료", "Lot"].astype(str).tolist()
    if not completed_lots:
        return pd.DataFrame(columns=columns)

    risk_source = snapshot_infer[snapshot_infer["Lot"].astype(str).isin(completed_lots)].copy()
    if risk_source.empty:
        return pd.DataFrame(columns=columns)
        
    def calc_severe_count(g, patience: int = 3):
        temp = g.sort_values("Datetime").copy()
        temp["is_anom"] = (temp["final_label"] == "ANOMALY").astype(int)

        if temp["is_anom"].sum() == 0:
            return 0

        temp["segment_start"] = (
            (temp["is_anom"] == 1) &
            (temp["is_anom"].shift(1, fill_value=0) == 0)
        ).astype(int)
        temp["anom_segment_id"] = temp["segment_start"].cumsum()

        anom = temp[temp["is_anom"] == 1].copy()
        if anom.empty:
            return 0

        seg_stats = anom.groupby("anom_segment_id").agg(
            seg_len=("combined_score", "size"),
            seg_max=("combined_score", "max")
        )

        return int(((seg_stats["seg_len"] >= patience) & (seg_stats["seg_max"] >= 5.7)).sum())

    severe_counts = risk_source.groupby("Lot").apply(calc_severe_count).reset_index(name="SevereAlertCount")

    risk_stats = (
        risk_source.sort_values("Datetime")
        .groupby("Lot")
        .agg(
            LatestScore=("combined_score", "last"),
            MaxScore=("combined_score", "max"),
            AlertEventCount=("is_daily_new_alert", "sum"),
            AnomalyRowCount=("final_label", lambda s: int((s == "ANOMALY").sum())),
            TotalScoredRows=("combined_score", "size"),
            LastAlertTime=("Datetime", lambda s: s[risk_source.loc[s.index, "is_daily_new_alert"]].max() if risk_source.loc[s.index, "is_daily_new_alert"].any() else pd.NaT),
        )
        .reset_index()
    )

    risk_stats = pd.merge(risk_stats, severe_counts, on="Lot", how="left")
    risk_stats["SevereAlertCount"] = risk_stats["SevereAlertCount"].fillna(0)

    risk_stats["AnomalyRatio"] = np.where(
        risk_stats["TotalScoredRows"] > 0,
        risk_stats["AnomalyRowCount"] / risk_stats["TotalScoredRows"],
        0.0,
    )
    risk_stats["Status"] = risk_stats.apply(
        lambda row: get_lot_risk_status(row["MaxScore"], row["AlertEventCount"], row["SevereAlertCount"]),
        axis=1,
    )
    risk_stats["LotRiskScore"] = (
        risk_stats["AlertEventCount"].astype(float) * 100.0
        + np.maximum(risk_stats["MaxScore"].astype(float) - 3.79, 0.0) * 10.0
    )

    risk_stats = risk_stats[
        (risk_stats["AlertEventCount"] > 0) | (risk_stats["AnomalyRowCount"] > 0)
    ].copy()

    return risk_stats.sort_values(
        by=["AlertEventCount", "MaxScore", "LatestScore"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_lot_board_df(
    full_day_raw: pd.DataFrame,
    snapshot_raw: pd.DataFrame,
    snapshot_infer: pd.DataFrame,
    threshold: float,
    window_size: int,
) -> pd.DataFrame:
    planned_lots = build_planned_lot_list(full_day_raw)
    snapshot_stats = build_snapshot_lot_stats(snapshot_infer, threshold)
    records = []

    for lot in planned_lots:
        raw_full_lot = full_day_raw[full_day_raw["Lot"] == lot].sort_values("Datetime")
        raw_snap_lot = snapshot_raw[snapshot_raw["Lot"] == lot].sort_values("Datetime")
        infer_snap_lot = snapshot_infer[snapshot_infer["Lot"] == lot].sort_values("Datetime")

        process_state = determine_process_state(raw_snap_lot, raw_full_lot, infer_snap_lot)
        raw_count = int(len(raw_snap_lot))
        score_ready = (not infer_snap_lot.empty) and (process_state == "완료")
        latest_score = float(infer_snap_lot["combined_score"].iloc[-1]) if score_ready else np.nan
        max_score = float(infer_snap_lot["combined_score"].max()) if score_ready else np.nan 
        risk_index = float(snapshot_stats.loc[lot, "RiskIndex"]) if score_ready and lot in snapshot_stats.index else 0.0
        risk_status = str(snapshot_stats.loc[lot, "Status"]) if score_ready and lot in snapshot_stats.index else process_state

        if process_state == "대기":
            bar_percent = 0.0
            sub_label = "예정"
            bar_color = process_state_color(process_state)

        elif process_state == "진행중":
            bar_percent = min(100.0, raw_count / max(len(raw_full_lot), 1) * 100.0)
            sub_label = "진행중"
            bar_color = process_state_color(process_state)

        else:  # 완료
            bar_percent = risk_index
            sub_label = risk_status
            
            if risk_status == "정상":
                bar_color = "#00cc66"
            elif risk_status == "주의":
                bar_color = "#ffcc00"
            elif risk_status == "이상":
                bar_color = "#ff4b4b"
            else:
                bar_color = risk_color(latest_score, threshold)

        planned_start = raw_full_lot["Datetime"].min() if not raw_full_lot.empty else pd.NaT
        latest_seen = raw_snap_lot["Datetime"].max() if not raw_snap_lot.empty else pd.NaT

        records.append(
            {
                "Lot": lot,
                "process_state": process_state,
                "risk_status": risk_status,
                "LatestScore": latest_score,
                "MaxScore": max_score,
                "RiskIndex": risk_index,
                "bar_percent": float(bar_percent),
                "bar_color": bar_color,
                "sub_label": sub_label,
                "raw_count": raw_count,
                "planned_start": planned_start,
                "latest_seen": latest_seen,
                "has_score": score_ready,
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "Lot", "process_state", "risk_status", "LatestScore", "MaxScore", "RiskIndex", "bar_percent",
                "bar_color", "sub_label", "raw_count", "planned_start", "latest_seen", "has_score"
            ]
        )

    board_df = pd.DataFrame(records)
    board_df = board_df.assign(_lot_sort_key=board_df["Lot"].map(natural_lot_key)).sort_values("_lot_sort_key").drop(columns=["_lot_sort_key"])
    return board_df.reset_index(drop=True)


def summarize_operating_lot(board_df: pd.DataFrame, snapshot_raw: pd.DataFrame, snapshot_infer: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    # 데이터가 아예 없으면 기본값 반환
    if snapshot_raw.empty: 
        return "-", pd.DataFrame()
        
    # 1. 무조건 현재 선택된 시점(snapshot)의 가장 마지막 데이터에 찍힌 로트 번호를 타겟으로 잡음
    latest_raw_row = snapshot_raw.sort_values("Datetime").iloc[-1]
    current_lot = str(latest_raw_row["Lot"])

    # 2. 해당 로트의 추론 데이터(점수)가 존재한다면 함께 반환하고, 이제 막 시작해서 점수가 없다면 빈 데이터프레임 반환
    if not snapshot_infer.empty and current_lot in snapshot_infer["Lot"].astype(str).unique().tolist():
        return current_lot, snapshot_infer[snapshot_infer["Lot"] == current_lot].copy()

    return current_lot, pd.DataFrame()




# =========================================================
# 공정 단계 시각화
# =========================================================
PROCESS_STEPS = ["탈지", "산세", "활성화", "도금", "크로메이트", "건조"]


def get_current_process_info(full_day_raw: pd.DataFrame, snapshot_raw: pd.DataFrame) -> dict:
    """대시보드의 분석 대상은 크로메이트 공정이므로, 공정 흐름에서 크로메이트 단계만 고정 강조한다."""
    chromate_step_idx = PROCESS_STEPS.index("크로메이트")

    if full_day_raw.empty or snapshot_raw.empty:
        return {
            "current_lot": "-",
            "current_step": "크로메이트",
            "current_step_idx": chromate_step_idx,
            "process_state": "대기",
        }

    latest_raw_row = snapshot_raw.sort_values("Datetime").iloc[-1]
    current_lot = str(latest_raw_row["Lot"])

    raw_full_lot = full_day_raw[full_day_raw["Lot"].astype(str) == current_lot].sort_values("Datetime")
    raw_snap_lot = snapshot_raw[snapshot_raw["Lot"].astype(str) == current_lot].sort_values("Datetime")

    if raw_full_lot.empty or raw_snap_lot.empty:
        process_state = "대기"
    elif len(raw_snap_lot) < len(raw_full_lot):
        process_state = "진행중"
    else:
        process_state = "완료"

    return {
        "current_lot": current_lot,
        "current_step": "크로메이트",
        "current_step_idx": chromate_step_idx,
        "process_state": process_state,
    }


def render_process_stepper(process_info: dict):
    step_html = []
    for i, step_name in enumerate(PROCESS_STEPS):
        classes = ["process-step"]

        if step_name == "크로메이트":
            classes.append("active")

        step_html.append(
            f'<div class="{" ".join(classes)}">'
            f'<div class="process-step-name">{step_name}</div>'
            f'</div>'
        )
        if i < len(PROCESS_STEPS) - 1:
            step_html.append('<div class="process-arrow">→</div>')

    process_html = (
        '<div class="process-stepper-wrap">'
        '<div class="process-flow">'
        f'{"".join(step_html)}'
        '</div>'
        '</div>'
    )
    st.markdown(process_html, unsafe_allow_html=True)


# =========================================================
# 3. 모델 / 아티팩트 로드
# =========================================================
def safe_load_keras_model(model_path: str):
    try:
        return load_model(model_path, compile=False)
    except TypeError as exc:
        if "quantization_config" not in str(exc):
            raise

        temp_path = None
        try:
            fd, temp_path = tempfile.mkstemp(suffix=".h5")
            os.close(fd)
            shutil.copyfile(model_path, temp_path)

            with h5py.File(temp_path, "r+") as h5_file:
                model_config = h5_file.attrs.get("model_config")
                if isinstance(model_config, bytes):
                    model_config = model_config.decode("utf-8")

                model_config = json.loads(model_config)

                def strip_quantization_config(node):
                    removed = 0
                    if isinstance(node, dict):
                        if "quantization_config" in node:
                            node.pop("quantization_config", None)
                            removed += 1
                        for value in node.values():
                            removed += strip_quantization_config(value)
                    elif isinstance(node, list):
                        for item in node:
                            removed += strip_quantization_config(item)
                    return removed

                removed_count = strip_quantization_config(model_config)
                if removed_count == 0:
                    raise

                h5_file.attrs.modify(
                    "model_config",
                    json.dumps(model_config).encode("utf-8")
                )

            return load_model(temp_path, compile=False)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


@st.cache_resource(show_spinner=True)
def load_model_and_artifacts(model_path: str, artifact_path: str):
    model = safe_load_keras_model(model_path)
    artifacts = joblib.load(artifact_path)
    return model, artifacts


@st.cache_data(show_spinner=True)
def load_raw_data(raw_data_path: str) -> pd.DataFrame:
    raw_df = pd.read_csv(raw_data_path)
    raw_df = ensure_datetime(raw_df)
    lot_col = determine_lot_col(raw_df)
    raw_df["Lot"] = raw_df[lot_col].astype(str)
    return raw_df


# =========================================================
# 4. 모델 추론
# =========================================================
def infer_one_lot(lot_df: pd.DataFrame, model, meta: dict, lot_col: str) -> pd.DataFrame:
    features = list(meta["FEATURES"])
    window_size = int(meta["WINDOW_SIZE"])

    g = lot_df.sort_values("Datetime").copy()
    if len(g) <= window_size:
        return pd.DataFrame()

    g = build_requested_features(g, features, group_col=lot_col)
    feature_arr = g[features].to_numpy(dtype=np.float32)

    scaler = meta.get("scaler") or meta.get("SCALER")
    if scaler is not None:
        feature_arr = scaler.transform(feature_arr).astype(np.float32)

    x_all, y_all, row_idx = make_sequences(feature_arr, window_size)
    if len(x_all) == 0:
        return pd.DataFrame()

    recon_pred, forecast_pred = model.predict(x_all, verbose=0)
    ae_score = np.mean((recon_pred[:, -1, :] - x_all[:, -1, :]) ** 2, axis=1)
    fc_score = np.mean((forecast_pred - y_all) ** 2, axis=1)

    ae_z = zscore_with_ref(meta["train_ae_score"], ae_score)
    fc_z = zscore_with_ref(meta["train_fc_score"], fc_score)

    combined_score = float(meta["AE_SCORE_WEIGHT"]) * ae_z + float(meta["FC_SCORE_WEIGHT"]) * fc_z
    threshold = float(meta["best_threshold"])

    out = g.iloc[row_idx].copy()
    out["ae_score"] = ae_score
    out["fc_score"] = fc_score
    out["combined_score"] = combined_score
    out["final_label"] = np.where(out["combined_score"] >= threshold, "ANOMALY", "NORMAL")
    out["normality_votes"] = out["combined_score"].apply(lambda x: score_to_votes(float(x), threshold))
    out["normal_votes"] = out["normality_votes"]
    out["risk_excess_score"] = np.maximum(out["combined_score"] - threshold, 0)
    out["RiskScore"] = out["risk_excess_score"]
    return out


@st.cache_data(show_spinner=True)
def load_and_infer_all(raw_data_path: str, model_path: str, artifact_path: str) -> pd.DataFrame:
    raw_df = pd.read_csv(raw_data_path)
    raw_df = ensure_datetime(raw_df)
    lot_col = determine_lot_col(raw_df)

    model, artifacts = load_model_and_artifacts(model_path, artifact_path)

    results = []
    grouped = raw_df.sort_values([lot_col, "Datetime"]).groupby(lot_col, sort=True)
    for _, g in grouped:
        pred_df = infer_one_lot(g, model=model, meta=artifacts, lot_col=lot_col)
        if not pred_df.empty:
            results.append(pred_df)

    if not results:
        return pd.DataFrame()

    data = pd.concat(results, ignore_index=True)
    data["Lot"] = data[lot_col].astype(str) if lot_col in data.columns else data["Lot"].astype(str)
    data = add_new_alert_flags(data, group_col="Lot")
    data = add_daily_alert_flags(data, group_col="Lot")
    return data


# =========================================================
# 5. 데이터 준비
# =========================================================
try:
    model, meta = load_model_and_artifacts(MODEL_PATH, ARTIFACT_PATH)
    threshold = float(meta["best_threshold"])
    window_size = int(meta["WINDOW_SIZE"])
    raw_data = load_raw_data(RAW_DATA_PATH)
    data = load_and_infer_all(RAW_DATA_PATH, MODEL_PATH, ARTIFACT_PATH)
except Exception as e:
    st.error(f"모델/아티팩트/데이터 로드 실패: {e}")
    st.stop()

if raw_data.empty:
    st.warning("원본 데이터가 없습니다.")
    st.stop()

# =========================================================
# 상단 필터 영역
# =========================================================

available_dates = sorted(raw_data["Datetime"].dt.date.unique(), reverse=True)
latest_date = max(available_dates)

# 최초 진입 시에만 메인에서 넘어온 기준 일자 사용
linked_date = st.session_state.get("linked_selected_date", None)

if "selected_date_calendar" not in st.session_state:
    if linked_date in available_dates:
        st.session_state["selected_date_calendar"] = linked_date
    else:
        st.session_state["selected_date_calendar"] = latest_date

if st.session_state["selected_date_calendar"] not in available_dates:
    st.session_state["selected_date_calendar"] = latest_date

# 날짜를 먼저 확정
filter_col1, filter_col2, top_spacer, filter_col3 = st.columns([1.0, 1.0, 2.0, 1.0])

with filter_col1:
    selected_date = st.date_input(
        "기준 일자",
        min_value=min(available_dates),
        max_value=max(available_dates),
        key="selected_date_calendar",
    )

# 날짜 확정 후 데이터 생성
full_day_raw = raw_data[
    raw_data["Datetime"].dt.date == selected_date
].copy().sort_values("Datetime")

if full_day_raw.empty:
    st.warning("선택한 날짜에 원본 데이터가 없습니다.")
    st.stop()

total_steps = len(full_day_raw)
slider_key = f"stream_idx_{selected_date}"
realtime_key = f"realtime_mode_{selected_date}"

planned_lots = build_planned_lot_list(full_day_raw)
lot_options = ["전체"] + planned_lots

if "selected_lot" not in st.session_state or st.session_state["selected_lot"] not in lot_options:
    st.session_state["selected_lot"] = "전체"

if slider_key not in st.session_state:
    st.session_state[slider_key] = 1

st.sidebar.markdown("### 실시간 제어")

sidebar_toggle = getattr(st.sidebar, "toggle", None)
if callable(sidebar_toggle):
    realtime_mode = sidebar_toggle("실시간 모드", value=False, key=realtime_key)
else:
    realtime_mode = st.sidebar.checkbox("실시간 모드", value=False, key=realtime_key)

if realtime_mode:
    if HAS_AUTOREFRESH:
        st_autorefresh(interval=5000, key=f"autorefresh_{selected_date}")
        if st.session_state[slider_key] < total_steps:
            st.session_state[slider_key] += 1
    else:
        st.sidebar.info("pip install streamlit-autorefresh 설치 필요")

time_options = full_day_raw["Datetime"].dt.strftime("%H:%M:%S").tolist()

# 최초 진입 시에만 메인에서 넘어온 기준 시간 사용
linked_time = st.session_state.get("linked_selected_time", None)

if linked_time in time_options and slider_key not in st.session_state:
    default_idx = time_options.index(linked_time)
else:
    default_idx = min(max(st.session_state.get(slider_key, 1), 1), len(time_options)) - 1

selected_time = st.sidebar.select_slider(
    "현재 시점",
    options=time_options,
    value=time_options[default_idx],
)

current_idx = time_options.index(selected_time) + 1
st.session_state[slider_key] = current_idx

st.session_state["linked_selected_date"] = selected_date
st.session_state["linked_selected_time"] = selected_time
st.session_state["linked_selected_timestamp"] = pd.to_datetime(f"{selected_date} {selected_time}")

snapshot_raw, selected_snapshot, snapshot_label, dashboard_mode = build_index_snapshot(
    full_day_raw=full_day_raw,
    current_idx=current_idx,
)

snapshot_data = data[
    (data["Datetime"].dt.date == selected_date) &
    (data["Datetime"] <= selected_snapshot)
].copy()

with filter_col2:
    st.text_input("기준 시간", value=snapshot_label, disabled=True)

with filter_col3:
    st.text_input("운영 모드", value=dashboard_mode, disabled=True)

current_process_info = get_current_process_info(full_day_raw, snapshot_raw)
render_process_stepper(current_process_info)

board_df = build_lot_board_df(full_day_raw, snapshot_raw, snapshot_data, threshold, window_size)
snapshot_lot_stats = build_snapshot_lot_stats(snapshot_data, threshold)
lot_risk_stats = build_lot_risk_stats(snapshot_data, board_df, threshold)

# =====================================================================
# 실시간 모드 ON일 때 '현재 로트' 자동 추적 (Auto-Tracking)
# =====================================================================
if realtime_mode and not snapshot_raw.empty:
    # 가장 마지막에 찍힌 원본 데이터의 로트 번호를 무조건 따라감
    live_lot = str(snapshot_raw.sort_values("Datetime").iloc[-1]["Lot"])
    if live_lot in lot_options:
        st.session_state["selected_lot"] = live_lot
        st.session_state["selected_lot_chart_filter"] = live_lot

# =========================================================
# KPI 섹션
# =========================================================
st.write("")
kpi_cols = st.columns(5)

started_lot_count = int(snapshot_raw["Lot"].nunique()) if not snapshot_raw.empty else 0
planned_lot_count = int(len(planned_lots))
alert_rows = snapshot_data[snapshot_data["is_daily_new_alert"]].sort_values("Datetime").copy() if not snapshot_data.empty else pd.DataFrame()
anomaly_total = int(alert_rows.shape[0])
anomaly_row_count = int((snapshot_data["final_label"] == "ANOMALY").sum()) if not snapshot_data.empty else 0
anomaly_rate = (anomaly_row_count / len(snapshot_data) * 100.0) if len(snapshot_data) > 0 else 0.0

if not board_df.empty and (board_df["process_state"] == "진행중").any():
    current_running_lot = str(
        board_df[board_df["process_state"] == "진행중"]
        .sort_values("latest_seen")
        .iloc[-1]["Lot"]
    )
else:
    current_running_lot = "-"

if selected_date == latest_date:
    lot_kpi_label = "현재 진행 Lot"
    lot_kpi_value = f"Lot {current_running_lot}" if current_running_lot != "-" else "-"
else:
    lot_kpi_label = "마지막 완료 Lot"
    lot_kpi_value = (
        f"Lot {snapshot_raw.sort_values('Datetime').iloc[-1]['Lot']}"
        if not snapshot_raw.empty else "-"
    )

urgent_count = int((board_df["risk_status"] == "이상").sum()) if not board_df.empty else 0

kpi_cols[0].metric("투입 / 예정", f"{started_lot_count} / {planned_lot_count} Lots")
kpi_cols[1].metric("누적 이상 감지율", f"{anomaly_rate:.2f}%")
kpi_cols[2].metric("이상탐지 건수", f"{anomaly_total}건")
kpi_cols[3].metric(lot_kpi_label, lot_kpi_value)
kpi_cols[4].metric("긴급 조치 필요", f"{urgent_count}건")


st.markdown("<hr>", unsafe_allow_html=True)

# =========================================================
# 섹션 2: 메인 분석
# =========================================================
main_left, main_right = st.columns([1, 2], gap="large")
left_top_container = main_left.container()
left_bottom_container = main_left.container()
right_container = main_right.container()

with left_top_container:
    st.subheader("위험 로트 리스트")

    abnormal_lot_stats = lot_risk_stats[lot_risk_stats["Status"] == "이상"].copy() if not lot_risk_stats.empty else pd.DataFrame()

    if abnormal_lot_stats.empty:
        st.info("현재 이상으로 분류된 완료 Lot이 없습니다.")
    else:
        for i, (_, row) in enumerate(abnormal_lot_stats.head(5).iterrows()):
            lot = str(row["Lot"])
            if st.button(
                f"{i+1}위: Lot {lot}",
                key=f"risk_{lot}",
                use_container_width=True,
            ):
                st.session_state["selected_lot"] = lot
                st.session_state["selected_lot_chart_filter"] = lot
                st.rerun()

with left_bottom_container:
    status_summary = board_df["process_state"].value_counts().to_dict() if not board_df.empty else {}

    title_col, info_col = st.columns([1.1, 2.4])
    with title_col:
        st.subheader("로트 상태")
    with info_col:
        st.markdown("<div style='margin-top: 10px; text-align: right; color: #666; font-size: 12px;'>"
                    f"대기 {status_summary.get('대기', 0)} | 진행중 {status_summary.get('진행중', 0)} | 완료 {status_summary.get('완료', 0)}"
                    "</div>", unsafe_allow_html=True)

    cols = st.columns(5)
    for i, (_, row) in enumerate(board_df.iterrows()):
        lot_id = str(row["Lot"])
        process_state = str(row["process_state"])
        latest_score = row["LatestScore"]
        max_score = row.get("MaxScore", np.nan)
        
        signal_color = get_lot_signal_color(process_state, str(row["risk_status"]), latest_score, threshold)
        info_text = get_lot_signal_text(
            process_state=process_state,
            display_score=max_score,
            threshold=threshold,
            planned_start=row["planned_start"],
            raw_count=int(row["raw_count"]),
            window_size=window_size,
        )

        with cols[i % 5]:
            if st.button(f"Lot {lot_id}", key=f"grid_lot_{lot_id}", use_container_width=True):
                st.session_state["selected_lot"] = lot_id
                st.session_state["selected_lot_chart_filter"] = lot_id
                st.rerun()

            st.markdown(
                f"""
                <div style="margin-top: -8px; margin-bottom: 18px; display: flex; flex-direction: column; align-items: center; gap: 4px;">
                    <div style="width: 12px; height: 12px; border-radius: 50%; background-color: {signal_color}; box-shadow: 0 0 0 2px rgba(0,0,0,0.06), 0 0 8px {signal_color}55;"></div>
                    <div style="font-size: 11px; color: #777;">{info_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with right_container:
    chart_title_col, chart_filter_col = st.columns([3.2, 1.0])

    with chart_title_col:
        st.subheader(f"센서 시계열 (날짜: {selected_date} / 기준: {format_datetime_label(selected_snapshot)} / Lot: {st.session_state['selected_lot']})")

    with chart_filter_col:
        selected_lot = st.selectbox(
            "LOT 선택 필터",
            lot_options,
            key="selected_lot_chart_filter",
            on_change=sync_selected_lot_from_filter,
        )
        st.session_state["selected_lot"] = selected_lot

    is_all_selected = st.session_state["selected_lot"] == "전체"

    if is_all_selected:
        plot_df = snapshot_raw.copy()
        anom_df = snapshot_data[snapshot_data["final_label"] == "ANOMALY"].copy() if not snapshot_data.empty else pd.DataFrame()
    else:
        plot_df = snapshot_raw[snapshot_raw["Lot"] == st.session_state["selected_lot"]].copy()
        anom_df = snapshot_data[
            (snapshot_data["Lot"] == st.session_state["selected_lot"]) &
            (snapshot_data["final_label"] == "ANOMALY")
        ].copy() if not snapshot_data.empty else pd.DataFrame()

    if plot_df.empty:
        st.info("선택한 Lot은 아직 시작 전입니다. 로트 상태판에서는 '대기'로 표시됩니다.")
    else:
        sensors = safe_sensor_columns(plot_df)
        if not sensors:
            st.warning("시계열로 표시할 pH / Temp / Voltage 컬럼이 없습니다.")
        else:
            for sensor in sensors:
                if is_all_selected:
                    fig = px.line(
                        plot_df,
                        x="Datetime",
                        y=sensor,
                        color="Lot",
                        color_discrete_sequence=[SERIES_BASE_COLOR],
                    )
                    fig.update_traces(
                        line=dict(color=SERIES_BASE_COLOR, width=1.0),
                        opacity=ALL_LOT_SERIES_OPACITY,
                    )
                else:
                    fig = px.line(
                        plot_df,
                        x="Datetime",
                        y=sensor,
                        color_discrete_sequence=[SERIES_BASE_COLOR],
                    )
                    fig.update_traces(
                        line=dict(color=SERIES_BASE_COLOR, width=1.6),
                        opacity=SINGLE_LOT_SERIES_OPACITY,
                    )

                if not anom_df.empty and sensor in anom_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=anom_df["Datetime"],
                            y=anom_df[sensor],
                            mode="markers",
                            marker=dict(color="red", size=6, symbol="x"),
                            name="이상",
                        )
                    )

                fig.update_layout(
                    height=250,
                    margin=dict(l=0, r=0, t=20, b=10),
                    showlegend=False,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    hoverlabel=dict(font_size=18, font_family="Arial"),
                )
                fig.update_yaxes(showgrid=True, gridcolor="#eee")
                st.plotly_chart(fig, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =========================================================
# 섹션 3: 이벤트 로그 및 진단
# =========================================================
log_col, diag_col = st.columns([2, 1])

with log_col:
    st.subheader("최근 이상탐지 이벤트 로그")

    if snapshot_data.empty:
        recent_events = pd.DataFrame()
    else:
        event_source = snapshot_data[snapshot_data["is_daily_new_alert"]].copy()
        if st.session_state["selected_lot"] != "전체":
            event_source = event_source[event_source["Lot"] == st.session_state["selected_lot"]].copy()
        recent_events = event_source.sort_values("Datetime", ascending=False).head(6).copy()

    if not recent_events.empty:
        recent_events["시간"] = recent_events["Datetime"].dt.strftime("%H:%M:%S")
        root_cause_df = recent_events.apply(
            lambda row: pd.Series(
                build_event_root_cause_info(
                    event_row=row,
                    raw_df=raw_data,
                    snapshot_infer_df=snapshot_data,
                    model=model,
                    meta=meta,
                    top_n=3,
                )
            ),
            axis=1,
        )

        recent_events = pd.concat([recent_events, root_cause_df], axis=1)

        recent_events["위험 등급"] = recent_events["combined_score"].apply(
            lambda x: get_status_badge(float(x), threshold)
        )

        st.table(
            recent_events[
                ["시간", "Lot", "주요 원인 변수", "주요 원인 패턴", "위험 등급"]
            ].reset_index(drop=True)
        )

    else:
        if st.session_state["selected_lot"] == "전체":
            st.info("현재 기준까지 감지된 이상 내역이 없습니다.")
        else:
            st.info(f"Lot {st.session_state['selected_lot']}에 해당하는 이상 이벤트가 없습니다.")

with diag_col:
    st.subheader("상세 진단 지표")

    if st.session_state["selected_lot"] == "전체":
        focus_lot, focus_score_df = summarize_operating_lot(board_df, snapshot_raw, snapshot_data)
    else:
        focus_lot = st.session_state["selected_lot"]
        focus_score_df = snapshot_data[snapshot_data["Lot"] == focus_lot].copy() if not snapshot_data.empty else pd.DataFrame()

    if board_df.empty or focus_lot == "-":
        diag_html = """<div style="background-color: white; border-radius: 10px; padding: 16px; box-shadow: 0px 4px 10px rgba(0,0,0,0.05); border: 1px solid #e0e0e0;">
<div style="font-size: 13px; font-weight: bold; margin-bottom: 8px;">상세 진단 지표</div>
<div style="font-size: 12px; color: #666;">표시할 Lot이 없습니다.</div>
</div>"""
        st.markdown(diag_html, unsafe_allow_html=True)

    else:
        board_row = board_df[board_df["Lot"] == focus_lot]
        process_state = board_row["process_state"].iloc[0] if not board_row.empty else "대기"
        lot_status = board_row["risk_status"].iloc[0] if not board_row.empty else process_state

        alert_count = 0
        severe_count = 0
        max_score = np.nan
        latest_score = np.nan

        if focus_lot in snapshot_lot_stats.index:
            stat_row = snapshot_lot_stats.loc[focus_lot]
            alert_count = int(stat_row.get("AlertCount", 0))
            severe_count = int(stat_row.get("SevereAlertCount", 0))
            max_score = float(stat_row.get("MaxScore", np.nan))
            latest_score = float(stat_row.get("LatestScore", np.nan))

        if process_state == "대기":
            status_text = "대기"
            extra_text = "아직 시작 전인 Lot입니다."
        elif process_state == "진행중":
            status_text = "진행중"
            extra_text = "실시간 수집 중인 Lot입니다."
        else:
            status_text = lot_status
            extra_text = "완료된 Lot의 누적 진단 정보입니다."

        # 상태별 텍스트 색상 매핑
        status_color = {
            "정상": "#00cc66",
            "주의": "#ffcc00",
            "이상": "#ff4b4b",
            "대기": "#b7b7b7",
            "진행중": "#5bc0de",
        }.get(status_text, "#333333")

        realtime_lines = ""
        if realtime_mode and not pd.isna(latest_score):
            realtime_lines = f"<div><b>현재 시점</b>: {format_datetime_label(selected_snapshot)}</div><div><b>현재 시점 점수</b>: {latest_score:.2f}</div>"

        max_score_text = "-" if pd.isna(max_score) else f"{max_score:.2f}"

        # 상태 텍스트 색상을 입힐 때 span 대신 strong 태그 사용!
        diag_html = f"""<div style="background-color: white; border-radius: 10px; padding: 18px; box-shadow: 0px 4px 10px rgba(0,0,0,0.05); border: 1px solid #e0e0e0;">
<div style="font-size: 22px; font-weight: 800; margin-bottom: 12px; color: #111;">Lot {focus_lot}</div>
<div style="font-size: 15px; line-height: 1.8; color: #222;">
<div><b>상태</b>: <strong style="color: {status_color};">{status_text}</strong></div>
<div><b>총 알림 횟수</b>: {alert_count}회</div>
<div><b>심각 알림 횟수</b>: {severe_count}회</div>
<div><b>Max score</b>: {max_score_text}</div>
{realtime_lines}
</div>
<div style="margin-top:10px; font-size:11px; color:#777; border-top: 1px solid #f0f0f0; padding-top: 8px;">{extra_text}</div>
</div>"""
        st.markdown(diag_html, unsafe_allow_html=True)


# =========================================================
# 섹션 4: ISO 규격 조치 이력 작성
# =========================================================

st.markdown("<hr>", unsafe_allow_html=True)




focus_lot_for_action = "-"
focus_status_for_action = "-"
focus_max_score = np.nan
focus_event_row = None
sensor_context = {}
active_alarms = []

selected_lot_for_action = st.session_state.get("selected_lot", "전체")

# 네 코드에서 이미 계산된 risk_status만 사용
action_candidate_df = board_df[
    (board_df["process_state"] == "완료") &
    (board_df["has_score"]) &
    (board_df["risk_status"].isin(["주의", "이상"]))
].copy() if not board_df.empty else pd.DataFrame()

# 이미 logs에 저장된 공정/Lot/상태는 조치 완료로 보고 제외
if not action_candidate_df.empty:
    action_candidate_df = action_candidate_df[
        ~action_candidate_df.apply(
            lambda row: is_action_completed(
                st.session_state.logs,
                "크로메이트",
                row["Lot"],
                row["risk_status"],
                selected_date
            ),
            axis=1
        )
    ].copy()

# 특정 Lot 선택 시: 그 Lot이 주의/이상이면 표시
if selected_lot_for_action != "전체":
    selected_board_row = action_candidate_df[
        action_candidate_df["Lot"].astype(str) == str(selected_lot_for_action)
    ].copy()

    if not selected_board_row.empty:
        row = selected_board_row.iloc[0]
        focus_lot_for_action = str(row["Lot"])
        focus_status_for_action = str(row["risk_status"])
        focus_max_score = float(row["MaxScore"])

# 전체 선택 시: 이상 우선, 그다음 MaxScore 높은 Lot 표시
else:
    if not action_candidate_df.empty:
        action_candidate_df["risk_priority"] = action_candidate_df["risk_status"].map({
            "이상": 2,
            "주의": 1
        })

        action_candidate_df = action_candidate_df.sort_values(
            ["risk_priority", "MaxScore", "LatestScore"],
            ascending=[False, False, False]
        )

        row = action_candidate_df.iloc[0]
        focus_lot_for_action = str(row["Lot"])
        focus_status_for_action = str(row["risk_status"])
        focus_max_score = float(row["MaxScore"])


# 이벤트 로그와 같은 함수로 원인 계산
if focus_lot_for_action != "-":
    focus_events = snapshot_data[
        (snapshot_data["Lot"].astype(str) == str(focus_lot_for_action)) &
        (snapshot_data["is_daily_new_alert"])
    ].copy()

    if not focus_events.empty:
        focus_event_row = focus_events.sort_values("Datetime", ascending=False).iloc[0]
    else:
        focus_lot_rows = snapshot_data[
            snapshot_data["Lot"].astype(str) == str(focus_lot_for_action)
        ].copy()

        if not focus_lot_rows.empty:
            focus_event_row = focus_lot_rows.sort_values("combined_score", ascending=False).iloc[0]

    if focus_event_row is not None:
        root_cause_info = build_event_root_cause_info(
            event_row=focus_event_row,
            raw_df=raw_data,
            snapshot_infer_df=snapshot_data,
            model=model,
            meta=meta,
            top_n=3,
        )

        dashboard_cause_variable = root_cause_info.get("주요 원인 변수", "-")
        dashboard_cause_pattern = root_cause_info.get("주요 원인 패턴", "-")

        ph_val = focus_event_row["pH"] if "pH" in focus_event_row.index and pd.notna(focus_event_row["pH"]) else None
        temp_val = focus_event_row["Temp"] if "Temp" in focus_event_row.index and pd.notna(focus_event_row["Temp"]) else None
        volt_val = focus_event_row["Voltage"] if "Voltage" in focus_event_row.index and pd.notna(focus_event_row["Voltage"]) else None

        sensor_context = {
            "process_name": "크로메이트",
            "lot": focus_lot_for_action,
            "status": focus_status_for_action,
            "max_score": round(focus_max_score, 2),
            "alarm_type": f"Lot {focus_lot_for_action} {focus_status_for_action}",
            "dashboard_cause_variable": dashboard_cause_variable,
            "dashboard_cause_pattern": dashboard_cause_pattern,
            "pH": ph_val,
            "Temp": temp_val,
            "Voltage": volt_val,
            "recent_actions": "최근 조치 이력 없음"
        }

        active_alarms.append(
            f"[시스템 경고] Lot {focus_lot_for_action} {focus_status_for_action} "
            f"(Max Score: {focus_max_score:.2f})"
        )


# 조치 UI
if active_alarms:
    alert_message = ", ".join(active_alarms)

    if focus_status_for_action == "주의":
        st.warning(alert_message)
        expander_title = f"[ISO 조치 이력 작성 - Lot {focus_lot_for_action} 주의]"
    else:
        st.error(alert_message)
        expander_title = f"[ISO 조치 이력 작성 - Lot {focus_lot_for_action} 이상]"

    action_text_key = f"chromate_action_text_{focus_lot_for_action}_{focus_status_for_action}"

    if action_text_key not in st.session_state:
        st.session_state[action_text_key] = ""

    with st.expander(expander_title, expanded=True):
        st.markdown(
            f"""
            - 주요 원인 변수: `{sensor_context.get("dashboard_cause_variable", "-")}`
            - 주요 원인 패턴: `{sensor_context.get("dashboard_cause_pattern", "-")}`
            """
        )

        if st.button(
            "AI 권고 조치 및 예방 방안 생성",
            key=f"chromate_ai_action_btn_{focus_lot_for_action}_{focus_status_for_action}"
        ):
            with st.spinner("분석 중."):
                result = generate_anomaly_report(sensor_context)

                if "error" not in result:
                    r = result.get("AI_Report", {})

                    draft_text = (
                        f"1. 원인: {sensor_context.get('dashboard_cause_variable', '-')}"
                        f" / {sensor_context.get('dashboard_cause_pattern', '-')}\n"
                        f"2. 권고 조치: {r.get('Corrective_Action', '')}\n"
                        f"3. 예방 방안: {r.get('Preventive_Action', '')}\n"
                        f"4. 실제 조치: "
                    )

                    st.session_state[action_text_key] = draft_text
                    st.rerun()
                else:
                    st.error(result["error"])

        with st.form(
            f"chromate_audit_form_{focus_lot_for_action}_{focus_status_for_action}",
            clear_on_submit=True
        ):
            st.markdown("##### 조치 결과 및 상황 전파")

            col1, col2 = st.columns(2)

            with col1:
                operator = st.selectbox(
                    "처리 담당자",
                    OPERATORS,
                    key=f"chromate_operator_{focus_lot_for_action}_{focus_status_for_action}"
                )

            with col2:
                notify_targets = st.multiselect(
                    "보고 대상",
                    NOTIFY_TARGETS,
                    default=["김윤환"],
                    key=f"chromate_notify_targets_{focus_lot_for_action}_{focus_status_for_action}"
                )

            action = st.text_area(
                "조치 내용",
                height=170,
                key=action_text_key
            )

            if st.form_submit_button("조치 업로드"):
                if action.strip():
                    report_to = ", ".join(notify_targets) if notify_targets else "보고 없음"

                    log_time = (
                        pd.Timestamp(selected_snapshot).strftime("%Y-%m-%d %H:%M:%S")
                        if pd.notna(selected_snapshot)
                        else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )

                    st.session_state.logs.insert(0, {
                        "일시": log_time,
                        "공정": "크로메이트",
                        "Lot": focus_lot_for_action,
                        "상태": focus_status_for_action,
                        "담당자": operator,
                        "보고 대상": report_to,
                        "내용": action
                    })


                    if notify_targets:
                        st.success(f"{report_to}에게 보고되었습니다. 조치 완료 처리되었습니다.")
                        time.sleep(1.5)
                    else:
                        st.success("조치 완료 처리되었습니다.")
                        time.sleep(1.0)

                    st.rerun()
                else:
                    st.warning("조치 내용을 입력하세요.")

else:
    if selected_lot_for_action == "전체":
        st.info("현재 조치가 필요한 주의/이상 Lot이 없거나 모두 조치 완료되었습니다.")
    else:
        st.info(f"Lot {selected_lot_for_action}은 현재 조치 대상이 아니거나 이미 조치 완료되었습니다.")


# 조치 이력
st.subheader("Lot별 공정 이상 조치 이력 (ISO Log)")

if st.session_state.logs:
    col1, col2 = st.columns([30, 2])

    with col2:
        if st.button("초기화", key="clear_logs_btn"):
            st.session_state.logs = []
            st.rerun()

    log_df = pd.DataFrame(st.session_state.logs)

    preferred_cols = ["일시", "공정", "Lot", "상태", "담당자", "보고 대상", "내용"]
    existing_cols = [col for col in preferred_cols if col in log_df.columns]

    st.dataframe(
        log_df[existing_cols],
        use_container_width=True
    )
else:
    st.info("아직 저장된 조치 이력이 없습니다.")
