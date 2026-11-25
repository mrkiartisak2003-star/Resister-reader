"""
app.py — หน้าเว็บเดโม: อัปโหลดรูปตัวต้านทาน แล้วแสดงค่าความต้านทาน

วิธีใช้:  streamlit run app.py
  เบราว์เซอร์จะเปิดหน้าเว็บขึ้นมาเอง (ถ้าไม่เปิด ให้เข้า http://localhost:8501)
  ปิดหน้าเว็บ: กลับมาที่ Terminal แล้วกด Control + C
"""
import glob
import os

import cv2
import joblib
import numpy as np
import streamlit as st

from color_code import draw_resistor, encode
from read_resistor import THAI, analyze, format_ohms

# สีที่ใช้แสดงแถบสีบนหน้าเว็บ
SWATCH = {"black": "#222222", "brown": "#7a3e12", "red": "#d32f2f", "orange": "#f57c00",
          "yellow": "#fdd835", "green": "#2e7d32", "blue": "#1565c0", "violet": "#7b1fa2",
          "grey": "#8d8d8d", "white": "#ffffff", "gold": "#c9a227", "silver": "#c0c0c0"}
LIGHT = {"yellow", "white", "gold", "silver", "orange"}   # สีอ่อน ใช้ตัวอักษรสีดำ

st.set_page_config(page_title="ระบบอ่านค่าตัวต้านทาน", page_icon="🔌", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("model/knn.joblib")


def chips_row(colors, confidence=None):
    chips = "".join(band_chip(i, c, None if confidence is None else confidence[i - 1])
                    for i, c in enumerate(colors, start=1))
    st.markdown(f"<div style='display:flex;gap:8px'>{chips}</div>", unsafe_allow_html=True)


def band_chip(i, color, conf):
    text_col = "#000" if color in LIGHT else "#fff"
    return (f"<div style='flex:1;text-align:center'>"
            f"<div style='background:{SWATCH[color]};color:{text_col};border:1px solid #999;"
            f"border-radius:8px;padding:14px 4px;font-size:20px;font-weight:600'>"
            f"{THAI[color]}</div>"
            f"<div style='font-size:13px;color:#888;margin-top:4px'>แถบ {i}"
            f"{'' if conf is None else f' · มั่นใจ {conf * 100:.0f}%'}</div>"
            f"</div>")


# ---------------------------------------------------------------------------
# แถบด้านข้าง
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🔌 เกี่ยวกับระบบ")
    st.markdown(
        "**ระบบอ่านค่าความต้านทานจากแถบสีบนภาพถ่ายตัวต้านทาน "
        "ด้วยวิธีสกัดคุณลักษณะสีในปริภูมิ HSV และ K-Nearest Neighbors**\n\n"
        "**ขั้นตอนการทำงาน**\n"
        "1. หาตัวต้านทานในภาพ (เทียบสีกับพื้นหลัง)\n"
        "2. หมุนให้วางแนวนอน\n"
        "3. แยกแถบสีจากโปรไฟล์สีตามแนวยาว\n"
        "4. **KNN** จำแนกสีของแต่ละแถบ\n"
        "5. แปลงรหัสสีเป็นค่าความต้านทาน\n\n"
        "**ข้อแนะนำการถ่ายภาพ**\n"
        "- วางบนกระดาษขาว ถ่ายจากด้านบน\n"
        "- แสงสม่ำเสมอ ไม่มีเงาพาด\n"
        "- ตัวต้านทานแบบ 4 แถบ รูปละ 1 ตัว"
    )

# ---------------------------------------------------------------------------
# หน้าหลัก: 2 แท็บ
# ---------------------------------------------------------------------------
DIGITS = {"black": 0, "brown": 1, "red": 2, "orange": 3, "yellow": 4,
          "green": 5, "blue": 6, "violet": 7, "grey": 8, "white": 9}


def explain(colors, value):
    mult = {"gold": "0.1", "silver": "0.01"}.get(colors[2]) or f"{10 ** DIGITS[colors[2]]:,}"
    st.markdown(f"<br>**วิธีคำนวณ:** ({DIGITS[colors[0]]}{DIGITS[colors[1]]}) × {mult} "
                f"= **{format_ohms(value)}**", unsafe_allow_html=True)


def tab_read():
    """แท็บ 1: อัปโหลดรูป -> ค่าโอห์ม"""
    st.caption("อัปโหลดรูปตัวต้านทาน 4 แถบ แล้วระบบจะบอกค่าความต้านทานให้")
    if not os.path.exists("model/knn.joblib"):
        st.error("ยังไม่มีโมเดล — ให้รัน `python train_knn.py` ใน Terminal ก่อน แล้วรีเฟรชหน้านี้")
        return
    bundle = load_model()

    source = st.radio("เลือกภาพจาก", ["อัปโหลดรูป", "ภาพตัวอย่างในโฟลเดอร์ images"], horizontal=True)
    img, name = None, None
    if source == "อัปโหลดรูป":
        upload = st.file_uploader("ลากรูปมาวาง หรือกดเลือกไฟล์", type=["jpg", "jpeg", "png"])
        if upload is not None:
            data = np.frombuffer(upload.getvalue(), np.uint8)
            img, name = cv2.imdecode(data, cv2.IMREAD_COLOR), upload.name
    else:
        samples = sorted(glob.glob("images/*.jpg") + glob.glob("images/*.png"))
        if samples:
            name = st.selectbox("เลือกภาพ", [os.path.basename(p) for p in samples])
            img = cv2.imread(os.path.join("images", name))
        else:
            st.info("ไม่มีภาพในโฟลเดอร์ images/")
    if img is None:
        if source == "อัปโหลดรูป":
            st.info("👆 อัปโหลดรูปเพื่อเริ่มอ่านค่า")
        return

    with st.spinner("กำลังวิเคราะห์ภาพ..."):
        res = analyze(img, bundle)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("ภาพที่วิเคราะห์")
        st.image(res["annotated"], channels="BGR", caption=name)
        if res["strip"] is not None:
            st.image(res["strip"], channels="BGR",
                     caption="ตัวต้านทานหลังหมุนให้ตรง พร้อมตำแหน่งแถบสีที่ตรวจพบ")
    with right:
        st.subheader("ผลการอ่านค่า")
        if res["ok"]:
            tol = f"±{res['tolerance']:g}%" if res["tolerance"] else "ไม่ทราบ"
            st.metric("ค่าความต้านทาน", format_ohms(res["value"]))
            st.metric("ค่าความคลาดเคลื่อน", tol)
        else:
            st.error(f"อ่านค่าไม่สำเร็จ: {res['error']}")
        if res["colors"]:
            st.markdown("**สีของแต่ละแถบ (ทายโดย KNN)**")
            chips_row(res["colors"], res["confidence"])
            if res["ok"]:
                explain(res["colors"], res["value"])


def tab_encode():
    """แท็บ 2: ใส่ค่าโอห์ม -> สีของแถบ"""
    st.caption("ใส่ค่าความต้านทาน แล้วระบบจะแสดงแถบสีของตัวต้านทานแบบ 4 แถบ")
    units = {"Ω": 1, "kΩ": 1e3, "MΩ": 1e6}
    tolerances = {"±5% (ทอง)": 5, "±10% (เงิน)": 10, "±1% (น้ำตาล)": 1, "±2% (แดง)": 2}
    c1, c2, c3 = st.columns([2, 1, 2])
    number = c1.number_input("ค่าความต้านทาน", min_value=0.0, value=4.7, step=0.1, format="%g")
    unit = c2.selectbox("หน่วย", list(units), index=1)
    tol_label = c3.selectbox("ค่าความคลาดเคลื่อน", list(tolerances))

    ohms = number * units[unit]
    try:
        colors, actual = encode(ohms, tolerances[tol_label])
    except ValueError as e:
        st.error(str(e))
        return

    if abs(actual - ohms) > 1e-9 * max(ohms, 1):
        st.warning(f"ตัวต้านทาน 4 แถบแสดงเลขได้แค่ 2 หลัก จึงปัด {format_ohms(ohms)} "
                   f"เป็น **{format_ohms(actual)}**")
    st.image(draw_resistor(colors), channels="BGR")
    st.metric("ค่าความต้านทาน", f"{format_ohms(actual)} {tol_label.split()[0]}")
    chips_row(colors)
    explain(colors, actual)


st.title("ระบบอ่านค่าตัวต้านทาน")
read_tab, encode_tab = st.tabs(["📷 อ่านค่าจากรูป", "🎨 หาสีจากค่าโอห์ม"])
with read_tab:
    tab_read()
with encode_tab:
    tab_encode()
