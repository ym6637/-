import os
import random
from datetime import datetime
from typing import Optional
from textwrap import dedent

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import font_manager, rcParams
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from streamlit_autorefresh import st_autorefresh
from streamlit_extras.stylable_container import stylable_container
USE_YOLO = os.getenv("ENABLE_YOLO", "0").strip().lower() in {"1", "true", "yes", "on"}

if USE_YOLO:
    try:
        from ultralytics import YOLO
        YOLO_IMPORT_ERROR = None
    except Exception as exc:
        YOLO = None
        YOLO_IMPORT_ERROR = exc
else:
    YOLO = None
    YOLO_IMPORT_ERROR = "YOLO disabled by default to keep Streamlit Cloud memory usage low."

try:
    import pillow_avif  # noqa: F401
except Exception:
    pillow_avif = None


# --------------------------
# 페이지 설정
# --------------------------
st.set_page_config(
    page_title="크로메이트 공정 불량 검출 대시보드",
    layout="wide",
)

# --------------------------
# 공통 설정값 및 경로
# --------------------------
CARD_HEIGHT = "1040px"
BOTTOM_HEIGHT = "360px"
HISTORY_PAGE_SIZE = 3
REFRESH_MS = 2000  # 2초마다 이미지 자동 전환
DISPLAY_LOT_COUNT = 22
LIVE_HISTORY_LIMIT = 60
LOT_HISTORY_LIMIT = 24
DISPLAY_IMAGE_MAX_SIDE = 1280
THUMBNAIL_MAX_SIDE = 320

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "converted_avif_lossless")
LOT_MAP_CSV = os.path.join(BASE_DIR, "LOT_IMAGE_MAP_CLEAN_READY.csv")
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
LOT_HISTORY_PAGE_SIZE = 5

CLASS_NAMES = {
    0: "plating",
    1: "scratch",
    2: "contamination",
    3: "pinhole",
}

CLASS_KR = {
    "plating": "도금불량",
    "scratch": "스크래치",
    "contamination": "표면오염",
    "pinhole": "핀홀",
}

CLASS_COLORS = {
    "plating": "#4e79ff",
    "scratch": "#ff6b6b",
    "contamination": "#ffb84d",
    "pinhole": "#4dd0e1",
}


# --------------------------
# 유틸 함수
# --------------------------
def set_korean_font():
    candidate_fonts = [
        "Malgun Gothic",
        "AppleGothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in candidate_fonts:
        if font_name in available:
            rcParams["font.family"] = font_name
            break
    rcParams["axes.unicode_minus"] = False


def calculate_recent_defect_rate(df, window=10):
    recent = df.head(window)
    prev = df.iloc[window:2 * window] if len(df) >= 2 * window else None

    if len(recent) == 0:
        return 0.0, "-", "데이터 없음", "-", "데이터 없음"

    recent_defects = (recent["Defect"] == 1).sum()
    recent_rate = (recent_defects / len(recent)) * 100

    if prev is None or len(prev) == 0:
        prev_rate = recent_rate
    else:
        prev_defects = (prev["Defect"] == 1).sum()
        prev_rate = (prev_defects / len(prev)) * 100

    if recent_rate >= 80:
        level_icon = "🚨"
        level = "위험"
    elif recent_rate >= 30:
        level_icon = "⚠"
        level = "경고"
    elif recent_rate >= 10:
        level_icon = "▲"
        level = "주의"
    elif recent_rate == 0:
        level_icon = "✓"
        level = "정상"
    else:
        level_icon = "●"
        level = "양호"

    if recent_rate > prev_rate:
        trend_icon = "↑"
        trend = "증가"
    elif recent_rate < prev_rate:
        trend_icon = "↓"
        trend = "감소"
    else:
        trend_icon = "→"
        trend = "유지"

    return recent_rate, level_icon, level, trend_icon, trend


def read_image_korean_path(image_path: str, max_side: int = DISPLAY_IMAGE_MAX_SIDE):
    try:
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((max_side, max_side))
        return np.array(image)
    except UnidentifiedImageError:
        return None
    except Exception:
        return None


def get_file_time_str(file_path: str):
    try:
        ts = os.path.getmtime(file_path)
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "-"


def infer_defect_types_from_filename(image_path: str):
    name = os.path.basename(image_path)
    mapping = {
        "도금": "plating",
        "스크래치": "scratch",
        "오염": "contamination",
        "핀홀": "pinhole",
    }
    defect_types = [eng for key, eng in mapping.items() if key in name]
    return sorted(defect_types)


def judge_status(boxes) -> str:
    return "불량" if len(boxes) > 0 else "정상"


def get_defect_types(boxes):
    return sorted(list({box["class_name"] for box in boxes}))


def recommend_action(boxes) -> str:
    if not boxes:
        return "현재 제품은 정상 판정입니다. 동일 조건을 유지하고 주기 점검만 수행하세요."

    defect_types = set(get_defect_types(boxes))
    actions = []

    if "scratch" in defect_types:
        actions.append("스크래치 검출: 이송 롤러, 가이드, 접촉부 마모 상태 우선 점검")
    if "contamination" in defect_types:
        actions.append("표면오염 검출: 세척 조건 및 이물 유입 여부 확인")
    if "pinhole" in defect_types:
        actions.append("핀홀 검출: 소재 표면 상태와 처리 조건 재점검")
    if "plating" in defect_types:
        actions.append("도금불량 검출: 도금 조건 및 전처리 상태 재확인")

    return " / ".join(actions)


def get_alarm_level(boxes) -> str:
    if not boxes:
        return "정상"
    defect_types = set(get_defect_types(boxes))
    if "plating" in defect_types or "pinhole" in defect_types:
        return "경고"
    return "주의"


def get_status_render_data(status: str, alarm_level: str):
    if status == "정상":
        return {
            "live_card_class": "card-live-normal",
            "status_html": "<div class='line-status-blink'><span class='status-dot-green'></span>라인 상태 정상</div>",
            "status_badge": "<span class='badge-normal'>정상</span>",
        }
    if alarm_level == "경고":
        return {
            "live_card_class": "card-live-alert",
            "status_html": "<div class='line-status-blink'><span class='status-dot-red'></span>즉시 점검 필요</div>",
            "status_badge": "<span class='badge-defect'>경고</span>",
        }
    return {
        "live_card_class": "card-live-warning",
        "status_html": "<div class='line-status-blink'><span class='status-dot-yellow'></span>주의 상태</div>",
        "status_badge": "<span class='badge-warning'>주의</span>",
    }


# --------------------------
# 모델 로드
# --------------------------
set_korean_font()
model = None
if YOLO is not None:
    try:
        model = YOLO(MODEL_PATH)
    except Exception as exc:
        YOLO_IMPORT_ERROR = exc

if model is not None:
    st.write("현재 모델:", model.ckpt_path)
else:
    st.info("배포 환경에서 YOLO/OpenCV를 불러오지 못해 파일명 기반 판정 모드로 실행 중입니다.")


@st.cache_data(show_spinner=False)
def predict_with_yolo(image_path: str):
    boxes_data = []
    if not os.path.exists(image_path):
        return boxes_data

    if model is None:
        return [
            {
                "class_name": defect_type,
                "conf": 1.0,
            }
            for defect_type in infer_defect_types_from_filename(image_path)
        ]

    if os.path.splitext(image_path)[1].lower() == ".avif":
        try:
            source = read_image_korean_path(image_path, max_side=960)
            if source is None:
                return boxes_data
        except UnidentifiedImageError:
            return boxes_data
        except Exception:
            return boxes_data
    else:
        source = image_path

    results = model(source, verbose=False)

    for result in results:
        if len(result.boxes) == 0:
            continue

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            x, y, w, h = box.xywhn[0].tolist()
            conf = float(box.conf[0].item())

            boxes_data.append(
                {
                    "class_id": cls_id,
                    "class_name": CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                    "conf": conf,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                }
            )
    return boxes_data


@st.cache_data(show_spinner=False)
def classify_image_status(image_path: str):
    boxes = predict_with_yolo(image_path)
    return judge_status(boxes)


@st.cache_data(show_spinner=False)
def get_prediction_bundle(image_path: str):
    boxes = predict_with_yolo(image_path)
    return {
        "boxes": boxes,
        "status": judge_status(boxes),
        "defect_types": get_defect_types(boxes),
        "action_text": recommend_action(boxes),
        "alarm_level": get_alarm_level(boxes),
        "selected_time": get_file_time_str(image_path),
    }


def prefetch_next_lot_prediction(image_files, current_idx: int):
    if not image_files:
        return
    next_idx = (current_idx + 1) % len(image_files)
    next_image = image_files[next_idx]
    next_img_path = os.path.join(IMAGE_DIR, next_image)
    get_prediction_bundle(next_img_path)


def build_stratified_shuffled_images(image_files, seed=42):
    normal_files = []
    defect_files = []

    for img_name in image_files:
        img_path = os.path.join(IMAGE_DIR, img_name)
        status = classify_image_status(img_path)
        if status == "불량":
            defect_files.append(img_name)
        else:
            normal_files.append(img_name)

    rng = random.Random(seed)
    rng.shuffle(normal_files)
    rng.shuffle(defect_files)

    mixed = []
    n_normal = len(normal_files)
    n_defect = len(defect_files)
    total = n_normal + n_defect

    if total == 0:
        return []

    target_normal_ratio = n_normal / total
    target_defect_ratio = n_defect / total

    used_normal = 0
    used_defect = 0

    while used_normal < n_normal or used_defect < n_defect:
        current_total = len(mixed)

        current_normal_ratio = (used_normal / current_total) if current_total > 0 else 0.0
        current_defect_ratio = (used_defect / current_total) if current_total > 0 else 0.0

        normal_gap = target_normal_ratio - current_normal_ratio if used_normal < n_normal else -999
        defect_gap = target_defect_ratio - current_defect_ratio if used_defect < n_defect else -999

        if normal_gap >= defect_gap and used_normal < n_normal:
            mixed.append(normal_files[used_normal])
            used_normal += 1
        elif used_defect < n_defect:
            mixed.append(defect_files[used_defect])
            used_defect += 1
        elif used_normal < n_normal:
            mixed.append(normal_files[used_normal])
            used_normal += 1

    return mixed


def draw_boxes(image, boxes):
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    h, w, _ = image.shape
    line_width = 6
    font_size = 30
    text_margin = 8
    text_padding_x = 8
    text_padding_y = 4

    try:
        font = ImageFont.truetype("malgunbd.ttf", font_size)
    except IOError:
        try:
            font = ImageFont.truetype("malgun.ttf", font_size)
        except IOError:
            try:
                font = ImageFont.truetype("AppleGothic.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()

    color_map = {
        "plating": (78, 121, 255),
        "scratch": (255, 107, 107),
        "contamination": (255, 184, 77),
        "pinhole": (77, 208, 225),
    }

    for box in boxes:
        if not all(key in box for key in ("x", "y", "w", "h")):
            continue
        x, y, bw, bh = box["x"], box["y"], box["w"], box["h"]
        name = box["class_name"]

        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)

        color = color_map.get(name, (0, 255, 0))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

        text = CLASS_KR.get(name, name)
        if "conf" in box:
            text = f"{text} {box['conf']:.2f}"

        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x1
        text_y = max(0, y1 - text_height - (text_padding_y * 2) - text_margin)

        draw.rectangle(
            (
                text_x,
                text_y,
                text_x + text_width + (text_padding_x * 2),
                text_y + text_height + (text_padding_y * 2),
            ),
            fill=color,
        )

        draw.text(
            (text_x + text_padding_x, text_y + text_padding_y - 1),
            text,
            font=font,
            fill=(255, 255, 255),
        )

    return np.array(img_pil)


@st.cache_data(show_spinner=False)
def build_log_rows(image_files):
    rows = []
    for img_name in image_files:
        img_path = os.path.join(IMAGE_DIR, img_name)
        boxes = predict_with_yolo(img_path)
        status = judge_status(boxes)
        defect_types = get_defect_types(boxes)

        rows.append(
            {
                "시간": get_file_time_str(img_path),
                "파일명": img_name,
                "판정": status,
                "결함": ", ".join([CLASS_KR.get(x, x) for x in defect_types]) if defect_types else "-",
            }
        )
    return rows
@st.cache_data(show_spinner=False)
def load_lot_map(csv_path: str):
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    for col in ["Lot", "anomaly_window_count", "max_combined_score", "severity_index"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required_cols = ["DATE_LOT", "lot_status", "assigned_image_name"]
    for col in required_cols:
        if col not in df.columns:
            return pd.DataFrame()

    df["assigned_image_name"] = df["assigned_image_name"].astype(str).str.strip()

    df = df[
        df["assigned_image_name"].notna()
        & (df["assigned_image_name"] != "")
        & (~df["assigned_image_name"].str.lower().eq("nan"))
        & (~df["assigned_image_name"].str.startswith("="))
    ].copy()

    df["lot_image_path"] = df["assigned_image_name"].apply(
        lambda x: os.path.join(IMAGE_DIR, x)
    )
    df["lot_image_exists"] = df["lot_image_path"].apply(os.path.exists)

    return df


def build_lot_rows(lot_df: pd.DataFrame):
    rows = []

    if lot_df.empty:
        return rows

    for _, row in lot_df.iterrows():
        image_name = row.get("assigned_image_name", "")
        image_path = row.get("lot_image_path", "")
        image_exists = bool(row.get("lot_image_exists", False))

        boxes = predict_with_yolo(image_path) if image_exists else []
        yolo_status = judge_status(boxes) if image_exists else "이미지없음"
        defect_types = get_defect_types(boxes) if image_exists else []

        rows.append(
            {
                "DATE_ONLY": row.get("DATE_ONLY", "-"),
                "Lot": row.get("Lot", "-"),
                "DATE_LOT": row.get("DATE_LOT", "-"),
                "lot_status": row.get("lot_status", "-"),
                "anomaly_window_count": row.get("anomaly_window_count", 0),
                "max_combined_score": row.get("max_combined_score", 0),
                "severity_bucket": row.get("severity_bucket", "-"),
                "assigned_image_name": image_name,
                "image_path": image_path,
                "image_exists": image_exists,
                "판정": yolo_status,
                "결함": ", ".join([CLASS_KR.get(x, x) for x in defect_types]) if defect_types else "-",
                "시간": get_file_time_str(image_path) if image_exists else "-",
            }
        )

    return rows


def build_balanced_lot_demo_data(image_names, target_count=DISPLAY_LOT_COUNT, seed=42):
    if not image_names:
        return [], {}, []

    shuffled = list(image_names)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    base_count = len(shuffled) // target_count
    remainder = len(shuffled) % target_count

    ordered_images = []
    image_to_lot = {}
    lot_rows = []
    cursor = 0

    for lot_no in range(1, target_count + 1):
        image_count = base_count + (1 if lot_no <= remainder else 0)
        lot_images = shuffled[cursor:cursor + image_count]
        cursor += image_count

        representative_image = lot_images[0] if lot_images else "-"
        representative_path = os.path.join(IMAGE_DIR, representative_image) if lot_images else ""

        if lot_images:
            # Keep runtime work bounded in deployment by using one representative
            # image per LOT instead of replaying the full folder on first load.
            ordered_images.append(representative_image)
            image_to_lot[representative_image] = lot_no

        lot_rows.append(
            {
                "DATE_ONLY": "-",
                "Lot": lot_no,
                "DATE_LOT": f"DEMO_LOT_{lot_no:02d}",
                "lot_status": "대기",
                "anomaly_window_count": 0,
                "max_combined_score": 0.0,
                "severity_bucket": "-",
                "assigned_image_name": representative_image,
                "image_path": representative_path,
                "image_exists": bool(lot_images) and os.path.exists(representative_path),
                "판정": "-",
                "결함": "-",
                "시간": get_file_time_str(representative_path) if lot_images and os.path.exists(representative_path) else "-",
                "image_count": len(lot_images),
            }
        )

    return ordered_images, image_to_lot, lot_rows


def lot_status_badge(status: str):
    if status == "이상":
        return "<span class='badge-defect'>이상</span>"
    if status == "주의":
        return "<span class='badge-warning'>주의</span>"
    return "<span class='badge-normal'>정상</span>"

def calc_defect_ratio(rows):
    counts = {
        "scratch": 0,
        "contamination": 0,
        "pinhole": 0,
        "plating": 0,
    }

    for row in rows:
        defect_types = row.get("defect_types")
        if defect_types is not None:
            for defect_type in defect_types:
                if defect_type in counts:
                    counts[defect_type] += 1
            continue

        defect_text = str(row.get("결함", ""))
        for eng, kr in CLASS_KR.items():
            if kr in defect_text:
                counts[eng] += 1

    total = sum(counts.values())
    ratio_rows = []
    for k, v in counts.items():
        pct = (v / total * 100) if total > 0 else 0
        ratio_rows.append(
            {
                "class_eng": k,
                "class_kr": CLASS_KR[k],
                "count": v,
                "pct": pct,
            }
        )
    return ratio_rows, total


def create_donut_chart_html(ratio_rows):
    nonzero = [r for r in ratio_rows if r["count"] > 0]

    if not nonzero:
        return """
        <div style="display:flex;justify-content:center;align-items:center;height:240px;
                    color:#667085;font-size:16px;font-weight:700;">
            집계 데이터 없음
        </div>
        """

    total = sum(r["count"] for r in nonzero)
    current = 0.0
    gradient_parts = []
    chart_label_parts = []
    side_legend_parts = []

    for row in nonzero:
        pct = (row["count"] / total) * 100
        start = current
        end = current + pct
        color = CLASS_COLORS[row["class_eng"]]
        gradient_parts.append(f"{color} {start:.2f}% {end:.2f}%")
        chart_label_parts.append(
            f"<div style='position:absolute;left:50%;top:50%;"
            f"transform:translate(-50%,-50%) rotate({(start + end) / 2 * 3.6:.2f}deg) "
            f"translateY(-97px) rotate(-{(start + end) / 2 * 3.6:.2f}deg);"
            f"color:#1f2430;font-size:12px;font-weight:800;'>{row['pct']:.0f}%</div>"
        )
        side_legend_parts.append(
            f"<div style='display:flex;align-items:center;gap:10px;padding:6px 0;'>"
            f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;"
            f"background:{color};flex:0 0 12px;'></span>"
            f"<span style='color:#344054;font-size:15px;font-weight:800;line-height:1.4;'>"
            f"{row['class_kr']}</span>"
            f"</div>"
        )
        current = end

    gradient = ", ".join(gradient_parts)
    labels_html = "".join(chart_label_parts)
    side_legend_html = "".join(side_legend_parts)

    return f"""
    <div style="display:flex;justify-content:flex-start;align-items:center;gap:24px;flex-wrap:nowrap;padding:8px 0 18px;overflow:hidden;">
        <div style="position:relative;width:236px;height:236px;flex:0 0 236px;">
            <div style="width:236px;height:236px;border-radius:50%;background:conic-gradient({gradient});"></div>
            <div style="position:absolute;inset:41px;border-radius:50%;background:#ffffff;
                        display:flex;flex-direction:column;align-items:center;justify-content:center;
                        box-shadow:inset 0 0 0 1px rgba(31,36,48,0.04);text-align:center;">
                <div style="color:#667085;font-size:15px;font-weight:800;">누적 결함</div>
                <div style="color:#1f2430;font-size:30px;font-weight:900;line-height:1.1;">{total}건</div>
            </div>
            {labels_html}
        </div>
        <div style="width:120px;min-width:120px;max-width:120px;flex:0 0 120px;">
            {side_legend_html}
        </div>
    </div>
    """


def create_donut_chart_html_with_label(ratio_rows, center_label: str = "누적 결함"):
    chart_html = create_donut_chart_html(ratio_rows)
    default_center_labels = [
        "?꾩쟻 寃고븿",
        "누적 결함",
    ]

    for default_label in default_center_labels:
        if default_label in chart_html:
            return chart_html.replace(default_label, center_label, 1)

    return chart_html


def sync_history_page_from_input():
    input_page = int(st.session_state.get("history_page_input", 1))
    total_pages = int(st.session_state.get("history_total_pages", 1))
    input_page = max(1, min(input_page, total_pages))
    st.session_state["history_page"] = input_page
    st.session_state["history_page_input"] = input_page


def go_to_prev_history_page():
    current_page = int(st.session_state.get("history_page", 1))
    new_page = max(1, current_page - 1)
    st.session_state["history_page"] = new_page
    st.session_state["history_page_input"] = new_page


def go_to_next_history_page():
    current_page = int(st.session_state.get("history_page", 1))
    total_pages = int(st.session_state.get("history_total_pages", 1))
    new_page = min(total_pages, current_page + 1)
    st.session_state["history_page"] = new_page
    st.session_state["history_page_input"] = new_page


def render_subsection_title(title: str, badge_html: str = ""):
    st.markdown(
        f"""
        <div style="
            display:flex;
            align-items:center;
            gap:8px;
            margin:4px 0 10px 0;
        ">
            <div style="
                font-size:18px;
                font-weight:900;
                color:#0f1b34;
            ">{title}</div>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_box(label: str, value: str):
    st.markdown(
        f"""
        <div style="
            background:#f8f9fc;
            border:1px solid rgba(31,36,48,0.06);
            border-radius:12px;
            min-height:78px;
            padding:12px 10px;
            text-align:center;
            display:flex;
            flex-direction:column;
            justify-content:center;
        ">
            <div style="
                font-size:12px;
                color:#667085;
                font-weight:800;
                margin-bottom:6px;
            ">{label}</div>
            <div style="
                font-size:17px;
                color:#344054;
                font-weight:900;
                line-height:1.35;
            ">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def merged_scroll_card_style(min_height: str = CARD_HEIGHT) -> str:
    return f"""
    {{
        background: #ffffff;
        border: 1px solid rgba(31, 36, 48, 0.08);
        border-radius: 18px;
        padding: 14px;
        height: {min_height};
        min-height: {min_height};
        max-height: {min_height};
        width: 100%;
        box-sizing: border-box;
        overflow-x: hidden;
        overflow-y: auto;
    }}
    """


def card_style(min_height: str = CARD_HEIGHT) -> str:
    return f"""
    {{
        background: #ffffff;
        border: 1px solid rgba(31, 36, 48, 0.08);
        border-radius: 18px;
        padding: 14px;
        height: {min_height};
        min-height: {min_height};
        max-height: {min_height};
        width: 100%;
        box-sizing: border-box;
        overflow: hidden;
    }}
    """


def action_card_style(min_height: str = BOTTOM_HEIGHT) -> str:
    return f"""
    {{
        background: #ffffff;
        border: 1px solid rgba(31, 36, 48, 0.08);
        border-radius: 18px;
        padding: 18px;
        min-height: {min_height};
        width: 100%;
        overflow-y: auto;
    }}
    """


def live_card_style_by_status(min_height: str, live_card_class: str) -> str:
    extra = ""
    if live_card_class == "card-live-normal":
        extra = """
        border: 1px solid rgba(42, 186, 119, 0.30) !important;
        box-shadow: 0 0 0 1px rgba(42, 186, 119, 0.08), 0 10px 24px rgba(42, 186, 119, 0.08);
        """
    elif live_card_class == "card-live-warning":
        extra = """
        border: 1px solid rgba(255, 184, 77, 0.40) !important;
        box-shadow: 0 0 0 1px rgba(255, 184, 77, 0.10), 0 10px 24px rgba(255, 184, 77, 0.10);
        """
    elif live_card_class == "card-live-alert":
        extra = """
        border: 1px solid rgba(255, 84, 84, 0.42) !important;
        box-shadow: 0 0 0 1px rgba(255, 84, 84, 0.12), 0 10px 28px rgba(255, 84, 84, 0.12);
        """

    return f"""
    {{
        background: #ffffff;
        border-radius: 18px;
        padding: 14px;
        height: {min_height};
        min-height: {min_height};
        max-height: {min_height};
        width: 100%;
        box-sizing: border-box;
        overflow: hidden;
        {extra}
    }}
    """


# --------------------------
# CSS
# --------------------------
st.markdown(
    """
<style>
html, body, [class*="css"] {
    background-color: #F0EEF3;
    color: #1f2430;
}
.stApp {
    background: #F0EEF3;
}
[data-testid="stDecoration"] {
    display: none;
}
.block-container {
    padding-top: 2.2rem;
    padding-bottom: 1rem;
    max-width: 1620px;
}
section[data-testid="stSidebar"] {
    width: 370px !important;
    min-width: 370px !important;
    max-width: 370px !important;
    background: #F0EEF3;
    border-right: 1px solid rgba(31, 36, 48, 0.08);
}
div[data-testid="stMarkdownContainer"] p {
    font-size: 16px;
    line-height: 1.6;
}
label, .stSelectbox label, .stRadio label, .stNumberInput label {
    font-size: 16px !important;
    font-weight: 700 !important;
    color: #344054 !important;
}
.main-title {
    font-size: 38px;
    font-weight: 900;
    color: #0f1b34;
    margin-top: 8px;
    margin-bottom: 10px;
    line-height: 1.25;
    letter-spacing: -0.5px;
    padding-top: 4px;
}
.section-title {
    font-size: 22px;
    font-weight: 900;
    color: #0f1b34;
    margin-bottom: 10px;
    letter-spacing: -0.2px;
}
.section-title-row {
    display: inline-flex;
    align-items: center;
    gap: 10px;
}
.metric-card {
    background: #ffffff;
    border: 1px solid rgba(31, 36, 48, 0.08);
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.30);
    min-height: 118px;
}
.metric-label {
    font-size: 15px;
    color: #667085;
    margin-bottom: 6px;
    font-weight: 800;
}
.metric-value {
    font-size: 40px;
    font-weight: 900;
    color: #1f2430;
    line-height: 1.0;
}
.small-text {
    color: #344054;
    font-size: 20px;
    font-weight: 700;
    line-height: 1.5;
}
.info-value {
    font-size: 28px;
    font-weight: 900;
    color: #1f2430;
}
.info-label {
    font-size: 18px;
    color: #667085;
    margin-top: 12px;
    margin-bottom: 8px;
    font-weight: 900;
}
.info-label-row {
    display: flex;
    align-items: center;
    gap: 10px;
}
.badge-normal {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(42, 186, 119, 0.20);
    color: #68f0a3;
    font-size: 14px;
    font-weight: 900;
}
.badge-defect {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(255, 84, 84, 0.20);
    color: #ff8f8f;
    font-size: 14px;
    font-weight: 900;
}
.badge-live {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: #0FA15E;
    border: 1px solid rgba(31, 36, 48, 0.08);
    color: #ffffff;
    font-size: 14px;
    font-weight: 900;
}
.badge-warning {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(255, 184, 77, 0.20);
    color: #ffd27c;
    font-size: 14px;
    font-weight: 900;
}
.status-dot-green,
.status-dot-red,
.status-dot-yellow {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
}
.status-dot-green {
    background: #34e38a;
}
.status-dot-red {
    background: #ff6464;
}
.status-dot-yellow {
    background: #ffc44d;
}
.history-item-wrap {
    padding-bottom: 2px;
    margin-bottom: 2px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.action-title {
    color: #9a6700;
    font-size: 22px;
    font-weight: 900;
    margin-bottom: 12px;
}
.action-text {
    color: #344054;
    font-size: 16px;
    line-height: 1.75;
}
.alarm-box {
    padding: 10px 12px;
    border-radius: 12px;
    background: #ffffff;
    border: 1px solid rgba(255, 184, 77, 0.28);
    color: #9a6700;
    font-size: 14px;
    font-weight: 700;
    margin-top: 8px;
}
hr {
    border: none;
    border-top: 1px solid rgba(31,36,48,0.08);
    margin: 8px 0;
}
@keyframes ledBlink {
    0% { opacity: 1; }
    50% { opacity: 0.35; }
    100% { opacity: 1; }
}
.live-frame {
    animation: smoothFade 0.45s ease-in-out;
    border-radius: 14px;
    overflow: hidden;
}
.card-live-normal {
    border: 1px solid rgba(42, 186, 119, 0.30) !important;
    box-shadow: 0 0 0 1px rgba(42, 186, 119, 0.08), 0 10px 24px rgba(42, 186, 119, 0.08);
}
.card-live-warning {
    border: 1px solid rgba(255, 184, 77, 0.40) !important;
    box-shadow: 0 0 0 1px rgba(255, 184, 77, 0.10), 0 10px 24px rgba(255, 184, 77, 0.10);
}
.card-live-alert {
    border: 1px solid rgba(255, 84, 84, 0.42) !important;
    box-shadow: 0 0 0 1px rgba(255, 84, 84, 0.12), 0 10px 28px rgba(255, 84, 84, 0.12);
}
@keyframes smoothFade {
    0% {
        opacity: 0;
        transform: scale(0.97);
    }
    100% {
        opacity: 1;
        transform: scale(1);
    }
}
.live-top-status {
    display: flex;
    justify-content: flex-start;
    align-items: center;
    padding-top: 2px;
    padding-left: 4px;
}
.live-time-only {
    color: #475467;
    font-size: 17px;
    font-weight: 800;
    text-align: left;
    white-space: nowrap;
    letter-spacing: -0.2px;
}
.line-status-blink {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 20px;
    font-weight: 700;
    color: #344054;
}
.line-status-blink .status-dot-green,
.line-status-blink .status-dot-red,
.line-status-blink .status-dot-yellow {
    animation: ledBlink 1.2s infinite;
}
.mid-bottom-card {
    background: #f8f9fc;
    border: 1px solid rgba(31, 36, 48, 0.06);
    border-radius: 14px;
    padding: 14px 16px;
    box-sizing: border-box;
    min-height: 98px;
    height: 98px;
    width: 100%;
    margin-top: 10px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
}
.bottom-card-title {
    font-size: 14px;
    color: #667085;
    font-weight: 800;
    margin-bottom: 8px;
    text-align: center;
    width: 100%;
}
.bottom-card-value {
    font-size: 18px;
    color: #344054;
    font-weight: 900;
    line-height: 1.35;
    text-align: center;
}
.result-card-flex {
    display: flex;
    flex-direction: column;
    height: 100%;
}
.result-bottom-anchor {
    margin-top: auto;
}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------
# 데이터 준비
# --------------------------
if not os.path.exists(IMAGE_DIR):
    st.error(f"이미지 폴더를 찾을 수 없습니다: {IMAGE_DIR}")
    st.stop()

source_image_files = sorted(
    [
        f
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".avif"))
    ]
)

if not source_image_files:
    st.warning("이미지 폴더에 표시할 이미지가 없습니다.")
    st.stop()

current_signature = tuple(source_image_files)

if "lot_ordered_images" not in st.session_state:
    ordered_images, image_to_lot, lot_rows = build_balanced_lot_demo_data(
        source_image_files,
        target_count=DISPLAY_LOT_COUNT,
        seed=42,
    )
    st.session_state["lot_ordered_images"] = ordered_images
    st.session_state["image_to_lot"] = image_to_lot
    st.session_state["lot_rows"] = lot_rows
    st.session_state["image_signature"] = current_signature

elif st.session_state.get("image_signature") != current_signature:
    ordered_images, image_to_lot, lot_rows = build_balanced_lot_demo_data(
        source_image_files,
        target_count=DISPLAY_LOT_COUNT,
        seed=42,
    )
    st.session_state["lot_ordered_images"] = ordered_images
    st.session_state["image_to_lot"] = image_to_lot
    st.session_state["lot_rows"] = lot_rows
    st.session_state["image_signature"] = current_signature

image_files = st.session_state.get("lot_ordered_images", [])
image_to_lot = st.session_state.get("image_to_lot", {})

if not image_files:
    st.warning("표시할 이미지 목록이 없습니다.")
    st.stop()
# --------------------------
# LOT 표시 데이터 준비
# converted_avif_lossless를 22개 LOT에 랜덤/균등 매핑 후, LOT 순서대로 재생
# --------------------------
lot_rows = st.session_state.get("lot_rows", [])

# --------------------------
# 실시간 이미지 자동 재생 상태
# --------------------------
if "auto_play" not in st.session_state:
    st.session_state["auto_play"] = True

if "selected_idx" not in st.session_state:
    if "selected_image" in st.session_state and st.session_state["selected_image"] in image_files:
        st.session_state["selected_idx"] = image_files.index(st.session_state["selected_image"])
    else:
        st.session_state["selected_idx"] = 0

if st.session_state["auto_play"]:
    refresh_count = st_autorefresh(interval=REFRESH_MS, key="live_refresh")
    st.session_state["selected_idx"] = refresh_count % len(image_files)
    st.session_state["selected_image"] = image_files[st.session_state["selected_idx"]]
else:
    if st.session_state.get("selected_image") in image_files:
        st.session_state["selected_idx"] = image_files.index(st.session_state["selected_image"])

def update_live_history(selected_image: str, status: str, defect_types=None, current_lot_no=None, max_len: Optional[int] = None):
    if "live_history" not in st.session_state:
        st.session_state["live_history"] = []

    history = st.session_state["live_history"]
    defect_types = defect_types or []
    defect_text = ", ".join([CLASS_KR.get(x, x) for x in defect_types]) if defect_types else "-"

    # 같은 이미지가 연속 rerun으로 중복 적재되는 것 방지
    if len(history) == 0 or history[-1]["file"] != selected_image:
        history.append({
            "file": selected_image,
            "status": status,
            "defect_types": defect_types,
            "defect_text": defect_text,
            "defect": 1 if status == "불량" else 0,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "lot_no": current_lot_no,
        })

    if max_len is not None and len(history) > max_len:
        st.session_state["live_history"] = history[-max_len:]


def update_lot_history(current_lot_no, selected_image: str, status: str, defect_types=None, max_len: Optional[int] = None):
    if current_lot_no is None:
        st.session_state["lot_history"] = []
        st.session_state["lot_history_lot_no"] = None
        return

    previous_lot_no = st.session_state.get("lot_history_lot_no")
    if previous_lot_no != current_lot_no:
        st.session_state["lot_history"] = []
        st.session_state["lot_history_lot_no"] = current_lot_no

    if "lot_history" not in st.session_state:
        st.session_state["lot_history"] = []

    history = st.session_state["lot_history"]
    defect_types = defect_types or []
    defect_text = ", ".join([CLASS_KR.get(x, x) for x in defect_types]) if defect_types else "-"

    if len(history) == 0 or history[-1]["file"] != selected_image:
        history.append({
            "file": selected_image,
            "status": status,
            "defect_types": defect_types,
            "defect_text": defect_text,
            "defect": 1 if status == "遺덈웾" else 0,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "lot_no": current_lot_no,
        })

    if max_len is not None and len(history) > max_len:
        st.session_state["lot_history"] = history[-max_len:]


def calculate_live_defect_rate(window: int = 20, trend_threshold: float = 5.0):
    history = st.session_state.get("live_history", [])

    if not history:
        return 0.0, "✓", "정상", "→", "데이터 없음"

    recent = history[-window:]
    prev = history[-2 * window:-window] if len(history) >= 2 * window else []

    recent_rate = sum(x["defect"] for x in recent) / len(recent) * 100

    if prev:
        prev_rate = sum(x["defect"] for x in prev) / len(prev) * 100
    else:
        prev_rate = recent_rate

    # 상태 기준 완화
    if recent_rate >= 40:
        level_icon = "🚨"
        level = "위험"
    elif recent_rate >= 20:
        level_icon = "⚠"
        level = "경고"
    elif recent_rate >= 10:
        level_icon = "▲"
        level = "주의"
    elif recent_rate == 0:
        level_icon = "✓"
        level = "정상"
    else:
        level_icon = "●"
        level = "양호"

    # 추세 기준 완화
    diff = recent_rate - prev_rate
    if diff >= trend_threshold:
        trend_icon = "↑"
        trend = "증가"
    elif diff <= -trend_threshold:
        trend_icon = "↓"
        trend = "감소"
    else:
        trend_icon = "→"
        trend = "유지"

    return recent_rate, level_icon, level, trend_icon, trend

# --------------------------
# 상태 / 페이지 처리
# --------------------------
selected_image = st.session_state.get("selected_image", image_files[0])
if selected_image not in image_files:
    selected_image = image_files[0]

st.session_state["selected_image"] = selected_image
st.session_state["selected_idx"] = image_files.index(selected_image)

# --------------------------
# 선택 이미지 처리
# --------------------------
selected_img_path = os.path.join(IMAGE_DIR, selected_image)
prediction = get_prediction_bundle(selected_img_path)
boxes = prediction["boxes"]
status = prediction["status"]
defect_types = prediction["defect_types"]
action_text = prediction["action_text"]
alarm_level = prediction["alarm_level"]
selected_time = prediction["selected_time"]
current_lot_no = image_to_lot.get(selected_image)

prefetch_next_lot_prediction(image_files, st.session_state["selected_idx"])

update_live_history(
    selected_image,
    status,
    defect_types=defect_types,
    current_lot_no=current_lot_no,
    max_len=LIVE_HISTORY_LIMIT,
)
update_lot_history(
    current_lot_no,
    selected_image,
    status,
    defect_types=defect_types,
    max_len=LOT_HISTORY_LIMIT,
)
recent_rate, level_icon, level, trend_icon, trend = calculate_live_defect_rate(
    window=20,
    trend_threshold=5.0
)
live_history_rows = st.session_state.get("live_history", [])
lot_history_rows = st.session_state.get("lot_history", [])
total_count = len(live_history_rows)
defect_count = sum(1 for x in live_history_rows if x["status"] == "불량")
yield_rate = ((total_count - defect_count) / total_count * 100) if total_count else 0
ratio_rows, _ = calc_defect_ratio(live_history_rows)
lot_ratio_rows, _ = calc_defect_ratio(lot_history_rows)
top_defect_row = max(ratio_rows, key=lambda x: x["count"]) if ratio_rows else None
top_defect_text = (
    f"{top_defect_row['class_kr']}"
    if top_defect_row and top_defect_row["count"] > 0
    else "없음"
)

lot_chart_title = (
    f"LOT 결함 분포 ({current_lot_no}번 LOT)"
    if current_lot_no is not None
    else "LOT 결함 분포"
)

render_data = get_status_render_data(status, alarm_level)
live_card_class = render_data["live_card_class"]
status_html = render_data["status_html"]
status_badge = render_data["status_badge"]

image_bgr = read_image_korean_path(selected_img_path, max_side=DISPLAY_IMAGE_MAX_SIDE)
if image_bgr is None:
    st.error(f"이미지를 읽을 수 없습니다: {selected_img_path}")
    st.stop()

image_rgb = image_bgr
boxed_image = draw_boxes(image_rgb, boxes)

# --------------------------
# LOT 선택 처리
# --------------------------
lot_selected_row = None
lot_selected_boxes = []
lot_selected_boxed_image = None
lot_selected_status = "데이터없음"
lot_selected_main_label = "-"
lot_selected_action = "LOT 데이터가 없습니다."
lot_selected_alarm_level = "정상"

if lot_rows and current_lot_no is not None:
    st.session_state["lot_selected_idx"] = max(0, min(current_lot_no - 1, len(lot_rows) - 1))

    lot_selected_row = dict(lot_rows[st.session_state["lot_selected_idx"]])
    lot_selected_boxes = boxes
    lot_selected_boxed_image = boxed_image
    lot_selected_status = "이상" if status == "불량" else status
    lot_selected_alarm_level = alarm_level
    lot_selected_action = action_text
    lot_primary_defect = defect_types[0] if defect_types else None
    lot_selected_main_label = CLASS_KR.get(lot_primary_defect, "정상") if lot_primary_defect else "정상"

    max_conf = max((box["conf"] for box in boxes), default=0.0)
    if status == "불량":
        if max_conf >= 0.8:
            severity_bucket = "Critical"
        elif max_conf >= 0.6:
            severity_bucket = "High"
        elif max_conf >= 0.4:
            severity_bucket = "Medium"
        else:
            severity_bucket = "Low"
    else:
        severity_bucket = "Low"

    lot_selected_row["assigned_image_name"] = selected_image
    lot_selected_row["image_path"] = selected_img_path
    lot_selected_row["image_exists"] = True
    lot_selected_row["판정"] = status
    lot_selected_row["결함"] = ", ".join([CLASS_KR.get(x, x) for x in defect_types]) if defect_types else "-"
    lot_selected_row["시간"] = selected_time
    lot_selected_row["lot_status"] = lot_selected_status
    lot_selected_row["anomaly_window_count"] = len(boxes)
    lot_selected_row["max_combined_score"] = max_conf * 100
    lot_selected_row["severity_bucket"] = severity_bucket

current_lot_value = f"{lot_selected_row['Lot']} / {DISPLAY_LOT_COUNT} Lots" if lot_selected_row is not None else f"- / {DISPLAY_LOT_COUNT} Lots"

# --------------------------
# 상단 제목
# --------------------------
st.markdown('<div class="main-title">크로메이트 공정 불량 검출 대시보드</div>', unsafe_allow_html=True)

# --------------------------
# KPI
# --------------------------
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-label">현재 검사 LOT</div>
        <div class="metric-value">{current_lot_value}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-label">총 검사 수</div>
        <div class="metric-value">{total_count}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-label">불량 수</div>
        <div class="metric-value">{defect_count}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with k4:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-label">양품률</div>
        <div class="metric-value">{yield_rate:.2f}%</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# --------------------------
# 상단 3영역
# --------------------------
st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
left, mid, right = st.columns(3)

with left:
    st.markdown('<div class="section-title">이미지 보기</div>', unsafe_allow_html=True)

    merged_left_style = live_card_style_by_status(CARD_HEIGHT, live_card_class).replace(
        "overflow: hidden;",
        "overflow-x: hidden; overflow-y: auto;",
    )

    with stylable_container("merged_image_card", css_styles=merged_left_style):
        render_subsection_title('실시간 검사 이미지', '<span class="badge-live">● LIVE</span>')

        control1, control2 = st.columns([1.2, 1.0])

        with control1:
            auto_play = st.checkbox(
                "실시간 모니터링",
                value=st.session_state.get("auto_play", True),
                key="auto_play",
            )

            if not auto_play:
                manual_idx = (
                    image_files.index(st.session_state["selected_image"])
                    if st.session_state["selected_image"] in image_files
                    else 0
                )

                selected_image = st.selectbox(
                    "검사 이미지 선택",
                    image_files,
                    index=manual_idx,
                    key="selected_image_manual",
                )

                st.session_state["selected_image"] = selected_image
                st.session_state["selected_idx"] = image_files.index(selected_image)

                selected_img_path = os.path.join(IMAGE_DIR, selected_image)
                prediction = get_prediction_bundle(selected_img_path)
                boxes = prediction["boxes"]
                status = prediction["status"]
                defect_types = prediction["defect_types"]
                action_text = prediction["action_text"]
                alarm_level = prediction["alarm_level"]
                selected_time = prediction["selected_time"]

                render_data = get_status_render_data(status, alarm_level)
                status_html = render_data["status_html"]
                status_badge = render_data["status_badge"]

                image_bgr = read_image_korean_path(selected_img_path, max_side=DISPLAY_IMAGE_MAX_SIDE)
                if image_bgr is not None:
                    image_rgb = image_bgr
                    boxed_image = draw_boxes(image_rgb, boxes)

                prefetch_next_lot_prediction(image_files, st.session_state["selected_idx"])

        with control2:
            st.markdown(
                f"""
                <div class='live-top-status'>
                    <div class='live-time-only'>{selected_time}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:2px;'></div>", unsafe_allow_html=True)
        st.image(boxed_image, use_container_width=True)

with mid:
    st.markdown('<div class="section-title">검사 결과</div>', unsafe_allow_html=True)
    with stylable_container("result_card", css_styles=card_style(CARD_HEIGHT)):
        st.markdown("<div class='result-card-flex'>", unsafe_allow_html=True)

        main_label = CLASS_KR.get(defect_types[0], "정상") if defect_types else "정상"

        top1, top2 = st.columns([1, 1])
        with top1:
            st.markdown(f"<div class='info-label info-label-row'>현재 판정 {status_badge}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='info-value'>{main_label}</div>", unsafe_allow_html=True)

        with top2:
            st.markdown("<div class='info-label'>라인 상태</div>", unsafe_allow_html=True)
            st.markdown(status_html, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"<div class='info-label'>{lot_chart_title}</div>", unsafe_allow_html=True)
        st.markdown(
            create_donut_chart_html_with_label(lot_ratio_rows, center_label="LOT 결함"),
            unsafe_allow_html=True,
        )

        st.markdown("<div class='info-label'>누적 결함 분포</div>", unsafe_allow_html=True)
        st.markdown(
            create_donut_chart_html_with_label(ratio_rows, center_label="누적 결함"),
            unsafe_allow_html=True,
        )

        st.markdown("<div class='result-bottom-anchor'>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class='mid-bottom-card'>
                <div class='bottom-card-title'>주요 결함</div>
                <div class='bottom-card-value'>{top_defect_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-title">이력 보기</div>', unsafe_allow_html=True)

    with stylable_container("merged_history_card", css_styles=merged_scroll_card_style(CARD_HEIGHT)):
        render_subsection_title("최근 검사 이력")

        local_filter_status = st.radio(
            "이력 필터",
            ["전체", "정상", "불량"],
            index=["전체", "정상", "불량"].index(st.session_state.get("history_filter", "전체")),
            horizontal=True,
            key="history_filter",
            label_visibility="collapsed",
        )

        live_history_rows = list(reversed(st.session_state.get("live_history", [])))

        if local_filter_status == "전체":
            local_filtered_logs_all = live_history_rows
        else:
            local_filtered_logs_all = [x for x in live_history_rows if x["status"] == local_filter_status]

        local_total_pages = max(1, (len(local_filtered_logs_all) + HISTORY_PAGE_SIZE - 1) // HISTORY_PAGE_SIZE)
        st.session_state["history_total_pages"] = local_total_pages

        local_history_page = st.session_state.get("history_page", 1)
        local_history_page = max(1, min(local_history_page, local_total_pages))
        st.session_state["history_page"] = local_history_page
        st.session_state["history_page_input"] = local_history_page

        local_start_idx = (local_history_page - 1) * HISTORY_PAGE_SIZE
        local_end_idx = local_start_idx + HISTORY_PAGE_SIZE
        local_filtered_logs = local_filtered_logs_all[local_start_idx:local_end_idx]

        if not local_filtered_logs:
            st.info("조건에 맞는 이력이 없습니다.")
        else:
            for row in local_filtered_logs:
                st.markdown('<div class="history-item-wrap">', unsafe_allow_html=True)

                h1, h2 = st.columns([1, 2.2])
                img_thumb_path = os.path.join(IMAGE_DIR, row["file"])
                thumb = read_image_korean_path(img_thumb_path, max_side=THUMBNAIL_MAX_SIDE)

                with h1:
                    if thumb is not None:
                        st.image(thumb)

                with h2:
                    badge = "badge-normal" if row["status"] == "정상" else "badge-defect"
                    history_lot_no = row.get("lot_no", image_to_lot.get(row["file"]))
                    history_lot_text = f"{history_lot_no}" if history_lot_no is not None else "-"
                    st.markdown(f"<span class='{badge}'>{row['status']}</span>", unsafe_allow_html=True)
                    st.markdown(f"<div class='small-text'>LOT: {history_lot_text}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='small-text'>시간: {row['time']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='small-text'>결함: {row['defect_text']}</div>", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

            pager_left, pager_center, pager_right = st.columns([0.8, 2.6, 0.8])

            with pager_left:
                st.button(
                    "◀",
                    key="history_prev",
                    use_container_width=True,
                    disabled=local_history_page <= 1,
                    on_click=go_to_prev_history_page,
                )

            with pager_center:
                st.markdown(
                    f"<div class='small-text' style='text-align:center;padding-top:6px;'>페이지 {local_history_page} / {local_total_pages}</div>",
                    unsafe_allow_html=True,
                )
                st.number_input(
                    "페이지 직접 이동",
                    min_value=1,
                    max_value=local_total_pages,
                    step=1,
                    key="history_page_input",
                    label_visibility="collapsed",
                    on_change=sync_history_page_from_input,
                )

            with pager_right:
                st.button(
                    "▶",
                    key="history_next",
                    use_container_width=True,
                    disabled=local_history_page >= local_total_pages,
                    on_click=go_to_next_history_page,
                )

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------
# 결함 가이드 데이터
# --------------------------
DEFECT_GUIDE = {
    "plating": {
        "label": "도금불량",
        "severity": "high",
        "summary": "도금/변환피막 형성이 불균일하거나 피복이 부족한 상태입니다.",
        "causes": [
            "전처리(탈지·세척) 부족으로 표면 활성화가 충분하지 않음",
            "도금액/크로메이트 용액 농도, pH, 온도가 관리 범위를 벗어남",
            "전류·전압 또는 침지 시간이 불안정하여 피막 형성이 고르지 않음",
            "지그·접촉부 불량으로 전류 전달 또는 부품 노출 상태가 불균일함",
        ],
        "checks": [
            "라인 전류·전압 로그와 침지 시간 이력 확인",
            "욕조 농도, pH, 온도, 오염도(슬러지·금속이온) 점검",
            "전처리 세척수, 탈지 상태, 수세 상태 재확인",
            "지그 체결 상태와 접촉부 마모 여부 확인",
        ],
        "actions": [
            "관리 기준을 벗어난 욕 조건을 보정하고 샘플 재검사",
            "전처리 후 재처리 가능한 LOT는 재세척 후 재도금/재처리",
            "지그 접촉부 청소·교체 후 재가동",
            "동일 조건 LOT를 묶어 추가 검사 대상으로 분리",
        ],
    },
    "contamination": {
        "label": "표면오염",
        "severity": "medium",
        "summary": "표면 이물 또는 세척 불량으로 인해 피막 품질 저하 가능성이 있는 상태입니다.",
        "causes": [
            "오일, 분진, 잔류 약품, 금속 미립자 등이 표면에 남아 있음",
            "수세 부족 또는 세척수 오염으로 오염물이 재부착됨",
            "탱크·필터·작업 환경 청정도가 떨어짐",
            "부품 적재/보관 중 외부 오염이 유입됨",
        ],
        "checks": [
            "세척·수세 공정 시간과 세척수 교체 주기 확인",
            "탱크 내 슬러지, 필터 막힘, 노즐 오염 여부 점검",
            "작업대·이송부·보관 트레이 청결 상태 확인",
            "오염이 특정 시간대/특정 LOT에 집중되는지 추적",
        ],
        "actions": [
            "세척/수세 조건 강화 후 시험편 재확인",
            "필터·탱크 청소 및 세척수 교체",
            "작업장 청정 관리와 보관 커버링 강화",
            "오염 집중 구간 설비를 세척한 뒤 재가동",
        ],
    },
    "scratch": {
        "label": "스크래치",
        "severity": "medium",
        "summary": "이송 또는 취급 중 기계적 접촉으로 표면 손상이 발생한 상태입니다.",
        "causes": [
            "컨베이어, 롤러, 가이드, 지그와의 마찰 또는 간섭",
            "작업자 취급 중 금속 간 접촉 또는 적재 불량",
            "이송 정렬 불량으로 부품이 흔들리거나 긁힘",
            "공정 전/후 보관 트레이 또는 치공구 마모",
        ],
        "checks": [
            "컨베이어·롤러·가이드 마모 및 돌출부 확인",
            "부품 간 간격과 적재 방식 확인",
            "지그/트레이 보호재 손상 여부 점검",
            "스크래치 위치가 반복되는지 확인해 설비 간섭 지점 추적",
        ],
        "actions": [
            "마찰 부품 정렬 보정 및 마모 부품 교체",
            "작업자 취급 기준과 적재 기준 재교육",
            "보호 패드/완충재 적용 또는 교체",
            "반복 위치 스크래치는 해당 설비 구간 일시 정지 후 점검",
        ],
    },
    "pinhole": {
        "label": "핀홀",
        "severity": "high",
        "summary": "작은 점상 결함 또는 미세 공극이 발생해 내식성 저하 우려가 큰 상태입니다.",
        "causes": [
            "표면 오염 또는 잔류 기포로 인해 피막이 연속적으로 형성되지 않음",
            "도금/변환피막 두께 부족 또는 반응 조건 불안정",
            "소재 표면 결함 또는 전처리 불량",
            "욕 오염, 교반 불량, 과도한 국부 반응",
        ],
        "checks": [
            "전처리 후 표면 상태와 수막 상태 확인",
            "욕조 교반, 농도, pH, 온도, 침지 시간 재점검",
            "기포 발생 위치와 설비 내 유동 상태 확인",
            "동일 형상 제품에서 반복 발생하는지 확인",
        ],
        "actions": [
            "욕 조건 안정화 후 시험편 재검사",
            "전처리 강화 및 기포 발생 구간 유동 개선",
            "도금/처리 시간과 조건 재설정",
            "핀홀 LOT는 우선 격리 후 내식성 위험 기준에 따라 재처리 또는 보류",
        ],
    },
}

def get_action_guide(defect_type):
    if defect_type is None:
        return None
    return DEFECT_GUIDE.get(str(defect_type).lower())

def build_guide_summary_html(defect_type):
    guide = get_action_guide(defect_type)
    if guide is None:
        return None

    severity_label = "즉시 점검" if guide["severity"] == "high" else "우선 점검"
    severity_class = "badge-defect" if guide["severity"] == "high" else "badge-warning"

    top_checks = guide["checks"][:2]
    top_actions = guide["actions"][:3]

    checks_html = "".join([f"<li>{item}</li>" for item in top_checks])
    actions_html = "".join([f"<li>{item}</li>" for item in top_actions])

    return dedent(f"""
    <div style="border:1.5px solid rgba(15,27,52,0.14); border-radius:16px; padding:18px; margin-top:10px; background:#ffffff;">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
            <div style="font-size:20px; font-weight:900; color:#0f1b34;">{guide['label']}</div>
            <span class="{severity_class}">{severity_label}</span>
        </div>
        <div style="font-size:15px; color:#1f2430; font-weight:800; line-height:1.7; margin-bottom:10px;">
            {guide['summary']}
        </div>
        <div style="font-size:14px; color:#344054; font-weight:900; margin:8px 0 6px 0;">핵심 점검</div>
        <ul style="margin:0 0 8px 18px; color:#1f2430; line-height:1.7;">{checks_html}</ul>
        <div style="font-size:14px; color:#344054; font-weight:900; margin:8px 0 6px 0;">즉시 조치</div>
        <ul style="margin:0 0 0 18px; color:#1f2430; line-height:1.7;">{actions_html}</ul>
    </div>
    """).strip()

# --------------------------
# 사용에 필요한 입력값
# --------------------------
# lot_primary_defect: 예) "pinhole", "plating", "scratch", "contamination"
# lot_selected_row: 예) {"DATE_LOT": "2021-10-06_10", "Lot": 10}
# lot_selected_main_label: 예) "핀홀"
# lot_selected_status: 예) "불량"

# --------------------------
# 점검 조치사항 UI
# --------------------------
with st.container():
    st.markdown('<div class="section-title">점검 조치사항</div>', unsafe_allow_html=True)

    with stylable_container("action_card_single", css_styles=action_card_style("360px")):
        st.markdown('<div class="action-title">점검 조치 가이드</div>', unsafe_allow_html=True)

        if lot_primary_defect is None:
            st.info("현재 선택된 LOT 이미지에서 결함이 검출되지 않았습니다.")
        else:
            summary_html = build_guide_summary_html(lot_primary_defect)
            if summary_html:
                st.markdown(summary_html, unsafe_allow_html=True)
            else:
                st.warning("해당 결함에 대한 요약 가이드가 정의되지 않았습니다.")

            guide = get_action_guide(lot_primary_defect)
            if guide is not None:
                with st.expander("상세보기"):
                    st.markdown("**주요 원인**")
                    for item in guide["causes"]:
                        st.markdown(f"- {item}")

                    st.markdown("**추가 점검 포인트**")
                    for item in guide["checks"]:
                        st.markdown(f"- {item}")

                    st.markdown("**전체 조치 가이드**")
                    for item in guide["actions"]:
                        st.markdown(f"- {item}")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class='action-text'>
            현재 LOT: {lot_selected_row['DATE_LOT']} / Lot {lot_selected_row['Lot']}<br>
            주요 결함: {lot_selected_main_label}<br>
            상태: {lot_selected_status}
            </div>
            """,
            unsafe_allow_html=True,
        )
