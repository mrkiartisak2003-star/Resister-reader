"""
read_resistor.py — ใส่รูปตัวต้านทาน แล้วบอกค่าความต้านทาน (Ω)

วิธีใช้:
  python read_resistor.py images/IMG_001.jpg        อ่านรูปเดียว
  python read_resistor.py images/*.jpg              อ่านหลายรูป
ต้องรัน train_knn.py ก่อนหนึ่งครั้ง เพื่อให้มีไฟล์โมเดล model/knn.joblib
ภาพผลลัพธ์ที่เขียนค่าไว้บนภาพจะถูกบันทึกในโฟลเดอร์ results/
"""
import os
import sys

import cv2
import joblib
import numpy as np

from detect import band_features, find_bands, find_resistor, straighten

THAI = {"black": "ดำ", "brown": "น้ำตาล", "red": "แดง", "orange": "ส้ม",
        "yellow": "เหลือง", "green": "เขียว", "blue": "น้ำเงิน", "violet": "ม่วง",
        "grey": "เทา", "white": "ขาว", "gold": "ทอง", "silver": "เงิน"}

# ตารางรหัสสีตัวต้านทาน
DIGIT = {"black": 0, "brown": 1, "red": 2, "orange": 3, "yellow": 4,
         "green": 5, "blue": 6, "violet": 7, "grey": 8, "white": 9}
MULTIPLIER = {**{c: 10 ** d for c, d in DIGIT.items()}, "gold": 0.1, "silver": 0.01}
TOLERANCE = {"brown": 1, "red": 2, "green": 0.5, "blue": 0.25, "violet": 0.1,
             "gold": 5, "silver": 10}


def format_ohms(value):
    for limit, unit in ((1e6, "MΩ"), (1e3, "kΩ")):
        if value >= limit:
            return f"{value / limit:g} {unit}"
    return f"{value:g} Ω"


def decode(colors):
    """แปลงสี 4 แถบ -> (ค่าความต้านทาน, % ความคลาดเคลื่อน) หรือ None ถ้าถอดรหัสไม่ได้"""
    c1, c2, c3, c4 = colors
    if c1 not in DIGIT or c2 not in DIGIT or c3 not in MULTIPLIER:
        return None
    value = (DIGIT[c1] * 10 + DIGIT[c2]) * MULTIPLIER[c3]
    return round(value, 2), TOLERANCE.get(c4)


def analyze(img, bundle):
    """วิเคราะห์ภาพ 1 รูป คืนค่า dict ที่มีผลทุกขั้น (ใช้ทั้งใน Terminal และหน้าเว็บ app.py)"""
    out = {"ok": False, "colors": [], "error": None, "annotated": img.copy(), "strip": None,
           "value": None, "tolerance": None, "confidence": []}
    rect = find_resistor(img)
    if rect is None:
        out["error"] = "หาตัวต้านทานในภาพไม่เจอ"
        return out
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.drawContours(out["annotated"], [box], 0, (0, 200, 0), 4)

    body, _ = straighten(img, rect)
    bands, body = find_bands(body)
    strip = body.copy()
    for i, (s, e) in enumerate(bands, start=1):
        cv2.rectangle(strip, (s, 2), (e, strip.shape[0] - 3), (0, 255, 0), 2)
        cv2.putText(strip, str(i), (s + 2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    out["strip"] = strip
    if len(bands) != 4:
        out["error"] = f"พบ {len(bands)} แถบ (ต้องการ 4 แถบ)"
        return out

    # ให้ KNN ทายสีของแต่ละแถบ (ใช้ค่า L, a, b ซึ่งอยู่ตำแหน่ง 3-5 ของ band_features)
    feats = np.array([band_features(body, b)[3:6] for b in bands])
    model = bundle["model"]
    out["colors"] = list(model.predict(feats))
    out["confidence"] = list(model.predict_proba(feats).max(axis=1))
    result = decode(out["colors"])
    if result is None:
        out["error"] = "ลำดับสีไม่ถูกต้องตามรหัสสี"
        return out

    out["ok"] = True
    out["value"], out["tolerance"] = result
    text = format_ohms(out["value"]) + (f" +/-{out['tolerance']:g}%" if out["tolerance"] else "")
    cv2.putText(out["annotated"], text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 10)
    cv2.putText(out["annotated"], text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 200), 4)
    return out


def read_resistor(path, bundle):
    img = cv2.imread(path)
    if img is None:
        return None, "เปิดไฟล์ภาพไม่ได้"
    res = analyze(img, bundle)
    if not res["ok"]:
        return res["colors"] or None, res["error"]
    os.makedirs("results", exist_ok=True)
    cv2.imwrite(os.path.join("results", os.path.basename(path)), res["annotated"])
    return res["colors"], (res["value"], res["tolerance"])


def main():
    if len(sys.argv) < 2:
        print("วิธีใช้: python read_resistor.py <ไฟล์ภาพ>")
        return
    if not os.path.exists("model/knn.joblib"):
        print("ยังไม่มีโมเดล ให้รัน python train_knn.py ก่อน")
        return
    bundle = joblib.load("model/knn.joblib")

    for path in sys.argv[1:]:
        colors, result = read_resistor(path, bundle)
        print(f"\n📷 {os.path.basename(path)}")
        if colors:
            print("   แถบสี           : " + " ".join(THAI[c] for c in colors))
        if isinstance(result, tuple):
            value, tol = result
            tol_text = f" ±{tol:g}%" if tol else ""
            print(f"   ค่าความต้านทาน : {format_ohms(value)}{tol_text}")
        else:
            print(f"   ❌ {result}")


if __name__ == "__main__":
    main()
