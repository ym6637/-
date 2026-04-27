
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, time as dt_time
import json
import os
import shutil
import tempfile
import time
import tensorflow as tf
import h5py

# 외부 서비스 모듈 임포트 (가정)
from openai_service import generate_anomaly_report

# ==========================================
# 0. 페이지 및 기본 설정
# ==========================================
st.set_page_config(page_title="QA/QC 통합 모니터링", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color: #F0EEF3 !important;
}

[data-testid="stMetric"] {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* 기준 일자 selectbox / 기준 시각 input 공통 흰색 박스 */
div[data-baseweb="input"],
div[data-baseweb="select"] > div {
    background-color: white !important;
    border: 1px solid #cccccc !important;
    border-radius: 8px !important;
    min-height: 42px !important;
    box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.03) !important;
    box-sizing: border-box !important;
}

div[data-baseweb="base-input"] {
    background-color: white !important;
}

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

h1, h2, h3, p, span, label {
    color: black !important;
}
            
</style>
""", unsafe_allow_html=True)


OPERATORS = ["손준호", "최윤호", "손영민", "허가은", "신소망", "윤다빈"]
CSV_PATH = "kemp-abh-sensor-final1.csv"

if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'scenario_idx' not in st.session_state:
    st.session_state.scenario_idx = 0

@st.cache_resource
def load_anomaly_model():
    model_path = 'hybrid_model.h5'
    if not os.path.exists(model_path):
        return None

    try:
        return tf.keras.models.load_model(model_path, compile=False)
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

            return tf.keras.models.load_model(temp_path, compile=False)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
    return None

@st.cache_data
def load_chromate_csv(csv_path):
    if not os.path.exists(csv_path):
        return None, None

    df = pd.read_csv(csv_path)
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)

    # 컬럼명 표준화
    rename_map = {}
    if "Temp" in df.columns:
        rename_map["Temp"] = "온도(C)"
    if "Voltage" in df.columns:
        rename_map["Voltage"] = "전압(V)"
    df = df.rename(columns=rename_map)

    # Ref 범위는 데이터셋 기반으로 5~95 분위수 사용
    ref_ranges = {
        "pH": (
            round(float(df["pH"].quantile(0.05)), 2),
            round(float(df["pH"].quantile(0.95)), 2)
        ),
        "온도(C)": (
            round(float(df["온도(C)"].quantile(0.05)), 2),
            round(float(df["온도(C)"].quantile(0.95)), 2)
        ),
        "전압(V)": (
            round(float(df["전압(V)"].quantile(0.05)), 2),
            round(float(df["전압(V)"].quantile(0.95)), 2)
        )
    }

    return df, ref_ranges

def detect_trend(series):
    if len(series) < 2:
        return "변화 미미"
    delta = float(series.iloc[-1] - series.iloc[0])
    if delta > 0.1:
        return "상승 추세"
    elif delta < -0.1:
        return "하락 추세"
    return "변화 미미"

def build_chromate_context(row, ref_ranges, df):
    recent = df[df["Datetime"] <= row["Datetime"]].tail(12)
    trend_map = {
        "pH": detect_trend(recent["pH"]),
        "온도(C)": detect_trend(recent["온도(C)"]),
        "전압(V)": detect_trend(recent["전압(V)"])
    }

    alarm_fields = []
    for name in ["pH", "온도(C)", "전압(V)"]:
        val = float(row[name])
        low, high = ref_ranges[name]
        if val < low:
            alarm_fields.append((name, val, low, high, "하한 이탈", trend_map[name]))
        elif val > high:
            alarm_fields.append((name, val, low, high, "상한 이탈", trend_map[name]))

    if alarm_fields:
        main_alarm = alarm_fields[0]
        sensor_context = {
            "process_name": "크로메이트",
            "tank_volume": 1500,
            "alarm_type": f"{main_alarm[0]} {main_alarm[4]}",
            "current_value": round(main_alarm[1], 2),
            "unit": "pH" if main_alarm[0] == "pH" else ("C" if main_alarm[0] == "온도(C)" else "V"),
            "low_limit": main_alarm[2],
            "high_limit": main_alarm[3],
            "trend": main_alarm[5],
            "other_sensors_status": "기타 변수 확인 필요" if len(alarm_fields) > 1 else "기타 변수 정상",
            "recent_actions": "최근 조치 이력 없음"
        }
    else:
        sensor_context = {}

    return alarm_fields, sensor_context, trend_map

hybrid_model = load_anomaly_model()
chromate_df, chromate_ref_ranges = load_chromate_csv(CSV_PATH)

# ==========================================
# 1. 데이터 연산부
# ==========================================
processes = ["탈지", "산세", "활성화", "도금", "크로메이트", "건조"]
default_process_params = {
    "탈지": {"온도(C)": (55, 65), "농도(%)": (4, 6), "시간(s)": (300, 600)},
    "산세": {"산농도(%)": (12, 18), "온도(C)": (22, 28), "철분(g/L)": (10, 40)},
    "활성화": {"산농도(%)": (2, 4), "온도(C)": (20, 24), "시간(s)": (40, 50)},
    "도금": {"전류밀도(A)": (3, 4), "온도(C)": (45, 55), "pH": (3.8, 4.2)},
    "크로메이트": {"pH": (1.8, 2.2), "온도(C)": (28, 32), "전압(V)": (11, 13)},
    "건조": {"온도(C)": (85, 95), "습도(%)": (5, 15), "시간(min)": (12, 18)}
}

process_params = default_process_params.copy()
if chromate_ref_ranges is not None:
    process_params["크로메이트"] = chromate_ref_ranges

# 크로메이트는 CSV 실측값을 쓰므로 시나리오에서 제외
SCENARIOS = [
    {"proc": "건조", "name": "온도(C)", "val": 105.0, "trend": "지속적 온도 상승", "type": "온도 상한 이탈"},
    {"proc": "산세", "name": "산농도(%)", "val": 10.5, "trend": "농도 저하 추세", "type": "농도 하한 이탈"},
    {"proc": "None", "name": "None", "val": None, "trend": "None", "type": "None"}
]
current_scenario = SCENARIOS[st.session_state.scenario_idx]

# ==========================================
# 사이드바 레이아웃 자리 먼저 잡기
# ==========================================
with st.sidebar:
    st.set_option("client.showSidebarNavigation", False)

    iso_sidebar_box = st.container()

    st.divider()

    sim_sidebar_box = st.container()

    time_slider_sidebar_box = st.container()

# ==========================================
# 화면 상단 제목
# ==========================================
head_l, head_c, head_r = st.columns([1.2, 8, 1.2])

with head_l:
    st.empty()

with head_c:
    st.markdown(
        """
        <h1 style='text-align: center; margin: 0; white-space: nowrap;'>
            통합 공정 실시간 모니터링 시스템
        </h1>
        """,
        unsafe_allow_html=True
    )

with head_r:
    alarm_placeholder = st.empty()

# ==========================================
# 2. 날짜 및 시간 선택 (CSV 기준)
# ==========================================
selected_row = None
selected_timestamp = None
chromate_alarm_fields = []
chromate_trend_map = {}
sensor_context = {}

if chromate_df is not None and not chromate_df.empty:
    min_date = chromate_df["Datetime"].min().date()
    max_date = chromate_df["Datetime"].max().date()

    default_dt = chromate_df["Datetime"].iloc[0]

    if "selected_date" not in st.session_state:
        st.session_state.selected_date = default_dt.date()

    if "selected_time_str" not in st.session_state:
        st.session_state.selected_time_str = default_dt.strftime("%H:%M:%S")

    # 본문에는 날짜 선택 + 기준 시각 표시만 둠
    date_col, filter_col2, empty1, empty2, empty3 = st.columns(5)

    with date_col:
        selected_date = st.date_input(
            "기준 일자",
            value=st.session_state.selected_date,
            min_value=min_date,
            max_value=max_date
        )

    day_df = chromate_df[chromate_df["Datetime"].dt.date == selected_date].copy()

    if day_df.empty:
        selected_row = chromate_df.iloc[0]
        selected_timestamp = selected_row["Datetime"]
        snapshot_label = selected_timestamp.strftime("%H:%M:%S")

        with filter_col2:
            st.text_input(
                "기준 시각",
                value=snapshot_label,
                disabled=True
            )

    else:
        time_options = day_df["Datetime"].dt.strftime("%H:%M:%S").tolist()

        if st.session_state.selected_time_str in time_options:
            default_idx = time_options.index(st.session_state.selected_time_str)
        else:
            default_idx = 0
            st.session_state.selected_time_str = time_options[default_idx]

        # 실제 시간 선택은 사이드바 슬라이더로 이동
        with time_slider_sidebar_box:
            selected_time_str = st.select_slider(
                "현재 시점",
                options=time_options,
                value=time_options[default_idx]
            )

        st.session_state.selected_date = selected_date
        st.session_state.selected_time_str = selected_time_str

        selected_timestamp = pd.to_datetime(f"{selected_date} {selected_time_str}")
        selected_row = day_df[day_df["Datetime"] == selected_timestamp].iloc[0]

        snapshot_label = selected_timestamp.strftime("%H:%M:%S")

        # 본문에는 기준 시각만 읽기 전용으로 표시
        with filter_col2:
            st.text_input(
                "기준 시각",
                value=snapshot_label,
                disabled=True
            )

    chromate_alarm_fields, sensor_context, chromate_trend_map = build_chromate_context(
        selected_row, process_params["크로메이트"], chromate_df
    )

else:
    date_col, filter_col2, empty1, empty2 = st.columns(4)

    with date_col:
        selected_date = st.date_input("기준 일자", datetime.now().date())

    fallback_time = datetime.now().time().strftime("%H:%M:%S")

    with filter_col2:
        st.text_input(
            "기준 시각",
            value=fallback_time,
            disabled=True
        )

    st.warning("CSV 파일을 찾지 못해 크로메이트 값은 기본 Ref 기준으로 동작합니다.")

# ==========================================
# 3. 센서값/알람 생성
# ==========================================
active_alarms = []
current_sensor_values = {}
process_statuses = {proc: "정상" for proc in processes}

for proc in processes:
    current_sensor_values[proc] = {}

    for name, (low, high) in process_params[proc].items():
        if proc == "크로메이트" and selected_row is not None:
            val = float(selected_row[name])
            is_err = (val < low) or (val > high)

            if is_err:
                process_statuses[proc] = "이상"
                if name == "pH":
                    model_score = 0.0
                    if hybrid_model is not None:
                        dummy_sequence = np.random.normal(0, 1, (1, 20, 15))
                        recon_pred, forecast_pred = hybrid_model.predict(dummy_sequence, verbose=0)
                        ae_score = np.mean(np.square(recon_pred[:, -1, :] - dummy_sequence[:, -1, :]), axis=1)[0]
                        fc_score = np.mean(np.square(forecast_pred - dummy_sequence[:, -1, :]), axis=1)[0]
                        model_score = (ae_score * 0.5) + (fc_score * 0.5)
                    active_alarms.append(f"{proc} {name} 이상 ({val:.2f}) | Model Score: {model_score:.4f}")
                else:
                    active_alarms.append(f"{proc} {name} 이상 ({val:.2f})")

            current_sensor_values[proc][name] = {"val": val, "is_err": is_err}
            continue

        is_target_anomaly = (proc == current_scenario["proc"] and name == current_scenario["name"])

        if is_target_anomaly:
            val = current_scenario["val"]
            active_alarms.append(f"{proc} {name} 이상 ({val})")
            process_statuses[proc] = "이상"

            # 크로메이트 센서 컨텍스트가 없을 때만 시나리오 컨텍스트 사용
            if not sensor_context:
                sensor_context = {
                    "process_name": proc,
                    "tank_volume": 1500,
                    "alarm_type": current_scenario["type"],
                    "current_value": val,
                    "unit": name.split('(')[-1].replace(')', '') if '(' in name else "",
                    "low_limit": low,
                    "high_limit": high,
                    "trend": current_scenario["trend"],
                    "other_sensors_status": "기타 변수 정상",
                    "recent_actions": "최근 조치 이력 없음"
                }
        else:
            val = np.random.uniform(low + (high - low) * 0.1, high - (high - low) * 0.1)

        current_sensor_values[proc][name] = {"val": val, "is_err": is_target_anomaly}

current_active_alarms = len(active_alarms)
if current_active_alarms > 0:
    alarm_placeholder.markdown("""
    <div title="현재 확인되지 않은 공정 이상이 존재합니다." style="display: flex; justify-content: flex-end; align-items: center; height: 100%; padding-top: 20px; cursor: help;">
        <div style="width: 25px; height: 25px; background-color: #ff1744; border-radius: 50%; animation: pulse 1.5s infinite; box-shadow: 0 0 10px #ff1744;"></div>
        <span style="margin-left: 10px; font-weight: bold; color: #ff1744;">ALARM</span>
    </div>
    <style>
    @keyframes pulse {
        0% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(255, 23, 68, 0.7);
        }
        70% {
            transform: scale(1);
            box-shadow: 0 0 0 10px rgba(255, 23, 68, 0);
        }
        100% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(255, 23, 68, 0);
        }
    }
    </style>
    """, unsafe_allow_html=True)
else:
    alarm_placeholder.empty()

total_incidents = current_active_alarms + len(st.session_state.logs)
dynamic_yield = max(0.0, 99.8 - (total_incidents * 1.5))
dynamic_oee = max(0.0, 89.5 - (total_incidents * 2.1))

yield_delta = f"{-1.5 * current_active_alarms:.1f}%" if current_active_alarms > 0 else "0.0%"
oee_delta = f"{-2.1 * current_active_alarms:.1f}%" if current_active_alarms > 0 else "0.0%"

iso_9001_status = "주의 (Warning)" if current_active_alarms > 0 else "준수 (Satisfied)"
iso_9001_color = "#fff3cd" if current_active_alarms > 0 else "#d4edda"
iso_9001_text = "#856404" if current_active_alarms > 0 else "#155724"

iso_14001_status = "위험 (Danger)" if process_statuses.get("크로메이트") == "이상" else "준수 (Satisfied)"
iso_14001_color = "#f8d7da" if process_statuses.get("크로메이트") == "이상" else "#d4edda"
iso_14001_text = "#721c24" if process_statuses.get("크로메이트") == "이상" else "#155724"

# ==========================================
# 4. 화면 렌더링 시작
# ==========================================
with iso_sidebar_box:
    st.header("ISO Compliance")
    st.caption("실시간 규격 준수 상태")

    def iso_box(title, status, color, text_color, help_text):
        st.markdown(f"""
        <div style="background-color:{color}; color:{text_color}; padding:12px; border-radius:8px; border-left: 5px solid {text_color}; margin-bottom: 10px; position: relative;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <strong>{title}</strong>
                <span title="{help_text}" style="cursor: help; font-size: 0.9em; color: {text_color}; opacity: 0.7; border: 1px solid {text_color}; border-radius: 50%; width: 18px; height: 18px; display: flex; justify-content: center; align-items: center;">?</span>
            </div>
            <div style="margin-top: 4px;">{status}</div>
        </div>
        """, unsafe_allow_html=True)

    iso_box("ISO 9001 (품질)", iso_9001_status, iso_9001_color, iso_9001_text,
            "제품 및 서비스가 고객 요구사항과 법적 규제를 지속적으로 충족함을 보증하는 국제 표준입니다.")
    iso_box("ISO 14001 (환경)", iso_14001_status, iso_14001_color, iso_14001_text,
            "기업 활동이 환경에 미치는 악영향을 최소화하고 법규를 준수하는 관리 체계를 의미합니다.")
    iso_box("ISO 45001 (안전)", "준수 (Satisfied)", "#d4edda", "#155724",
            "산업재해를 예방하고 안전한 근로 환경을 제공하기 위한 국제 표준입니다.")


with sim_sidebar_box:
    st.header("실시간 제어")
    sim_on = st.toggle(
        "자동 공정 진행 (Auto-Run)",
        value=False,
        help="정상 상태일 때 5초마다 다음 에러 시나리오를 자동 발생시킵니다."
    )

# KPI 대시보드
st.subheader("핵심 성과 지표 (KPI)")
k1, k2, k3, k4 = st.columns(4)
k1.metric("양품률", f"{dynamic_yield:.1f}%", yield_delta, help="누적 불량 발생에 따른 양품 비율입니다.")
k2.metric("OEE", f"{dynamic_oee:.1f}%", oee_delta, help="설비 종합 효율 지표입니다.")
k3.metric("이상알람", f"{current_active_alarms}건", f"{current_active_alarms}건" if current_active_alarms > 0 else None, delta_color="inverse")
k4.metric("목표생산 달성률", "95.2%", "1.2%")

st.divider()

# 공정별 상태
st.subheader("공정별 탱크 실시간 상태")
try:
    st.image("image.png", use_container_width=True)
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

status_themes = {
    "정상": {"bg": "#d4edda", "color": "#155724", "border": "#c3e6cb"},
    "주의": {"bg": "#fff3cd", "color": "#856404", "border": "#ffeeba"},
    "이상": {"bg": "#f8d7da", "color": "#721c24", "border": "#f5c6cb"}
}

status_cols = st.columns(6)
for i, proc in enumerate(processes):
    status = process_statuses[proc]
    theme = status_themes[status]
    with status_cols[i]:
        st.markdown(
            f'<div style="background-color: {theme["bg"]}; color: {theme["color"]}; padding: 10px; border-radius: 5px; text-align: center; border: 2px solid {theme["border"]}; font-weight: bold; margin-bottom: 10px; margin-top: -15px;">{proc}<br>{status}</div>',
            unsafe_allow_html=True
        )

v_cols = st.columns(6)
for i, proc in enumerate(processes):
    with v_cols[i]:
        st.markdown(
            f"<div style='background-color: #343a40; color: white; padding: 8px; border-radius: 5px; text-align: center; font-weight: bold;'>{proc}</div>",
            unsafe_allow_html=True
        )
        for name, data in current_sensor_values[proc].items():
            low, high = process_params[proc][name]
            t_color = "#ff1744" if data["is_err"] else "#212529"
            st.markdown(
                f'<div style="background-color: #f8f9fa; padding: 10px; border: 1px solid #dee2e6; border-top: none; margin-bottom: 2px;">'
                f'<div style="font-size: 0.8em; color: #6c757d;">{name}</div>'
                f'<div style="font-size: 1.2em; font-weight: bold; color: {t_color};">{data["val"]:.2f}</div>'
                f'<div style="font-size: 0.7em; color: #adb5bd;">Ref: {low:.2f}-{high:.2f}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

process_pages = {
    "탈지": "pages/탈지.py",
    "산세": "pages/산세.py",
    "활성화": "pages/활성화.py",
    "도금": "pages/도금.py",
    "크로메이트": "pages/chromate.py",
    "건조": "pages/건조.py"
}

detail_cols = st.columns(6)

for i, proc in enumerate(processes):
    with detail_cols[i]:
        if st.button(
            f"{proc} 상세 페이지",
            key=f"go_detail_{proc}",
            use_container_width=True
        ):
            st.session_state["linked_process"] = proc
            st.session_state["linked_selected_date"] = selected_date
            st.session_state["linked_selected_time"] = st.session_state.selected_time_str
            st.session_state["linked_selected_timestamp"] = selected_timestamp

            st.switch_page(process_pages[proc])



st.subheader("공정 이상 조치 이력 (ISO Log)")

if st.session_state.logs:
    st.dataframe(
        pd.DataFrame(st.session_state.logs),
        use_container_width=True
    )
else:
    st.info("아직 저장된 조치 이력이 없습니다.")

# 자동 시뮬레이션은 비크로메이트 랜덤 시나리오만 진행
if sim_on and not active_alarms:
    with st.spinner("공정 정상 가동 중..."):
        time.sleep(5)
    st.session_state.scenario_idx = (st.session_state.scenario_idx + 1) % len(SCENARIOS)
    st.rerun()
