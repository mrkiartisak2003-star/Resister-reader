"""
detect.py — หาตัวต้านทานในภาพ แล้วแยกแถบสีออกมา

วิธีใช้:  python detect.py
  - อ่านภาพทุกไฟล์ในโฟลเดอร์ images/
  - บันทึกภาพผลลัพธ์ (ตีกรอบแถบสีที่เจอ) ไว้ในโฟลเดอร์ output/
  - บันทึกค่าสีของแต่ละแถบ (HSV และ Lab) ไว้ใน dataset/bands.csv
    ถ้ามีไฟล์ images/labels.csv จะใส่ชื่อสีที่ถูกต้องให้ด้วย (ใช้เทรน KNN)
"""
import csv
import glob
import os

import cv2
import numpy as np

IMAGE_DIR = "images"
OUTPUT_DIR = "output"
DATASET_DIR = "dataset"


# ---------------------------------------------------------------------------
# ขั้นที่ 1: หาตัวต้านทานในภาพ
# ---------------------------------------------------------------------------
def find_resistor(img):
    """คืนค่ากรอบสี่เหลี่ยมเอียง (minAreaRect) ของตัวต้านทาน หรือ None ถ้าไม่เจอ"""
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)

    # สีพื้นหลัง = ค่ากลางของพิกเซลรอบขอบภาพ (กระดาษขาว)
    border = np.concatenate([lab[:20].reshape(-1, 3), lab[-20:].reshape(-1, 3),
                             lab[:, :20].reshape(-1, 3), lab[:, -20:].reshape(-1, 3)])
    bg = np.median(border, axis=0)

    # พิกเซลที่สีต่างจากพื้นหลังมาก = วัตถุ (ตัวต้านทาน + ขา)
    diff = np.linalg.norm(lab - bg, axis=2)
    mask = (diff > 25).astype(np.uint8) * 255

    # ลบขาโลหะที่บาง ๆ ออก: ขนาด kernel ปรับตามความหนาของตัวต้านทาน
    # (หาความหนาจาก distance transform = ระยะจากขอบถึงกลางตัว)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    half_thick = cv2.distanceTransform(mask, cv2.DIST_L2, 5).max()
    k = max(9, int(half_thick * 0.9) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(cv2.contourArea(c) for c in contours)
    if biggest < 0.005 * img.shape[0] * img.shape[1]:
        return None
    # แถบสีขาว/เทาที่สีใกล้กระดาษอาจทำให้ตัวต้านทานขาดเป็นหลายชิ้น
    # จึงรวมทุกชิ้นที่ใหญ่พอ แล้วหากรอบที่ครอบทั้งหมด
    pieces = [c for c in contours if cv2.contourArea(c) > 0.05 * biggest]
    return cv2.minAreaRect(np.vstack(pieces))


# ---------------------------------------------------------------------------
# ขั้นที่ 2: หมุนให้ตัวต้านทานวางแนวนอน แล้วตัดเฉพาะส่วนตัว
# ---------------------------------------------------------------------------
def straighten(img, rect):
    (cx, cy), (w, h), angle = rect
    if w < h:                      # ให้ด้านยาวเป็นแนวนอนเสมอ
        w, h = h, w
        angle += 90
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    crop = cv2.getRectSubPix(rotated, (int(w), int(h)), (cx, cy))
    return crop, M


# ---------------------------------------------------------------------------
# ขั้นที่ 3: หาแถบสีจากโปรไฟล์สีตามแนวยาว
# ---------------------------------------------------------------------------
def find_bands(body):
    """คืนรายการช่วง (x_start, x_end) ของแต่ละแถบ เรียงจากแถบที่ 1 ถึงแถบสุดท้าย"""
    h, w = body.shape[:2]
    # ใช้เฉพาะแถบกลางของตัว (ตัดขอบบน-ล่างที่มีเงาและแสงสะท้อนออก)
    strip = body[int(h * 0.30): int(h * 0.70)]
    lab = cv2.cvtColor(strip, cv2.COLOR_BGR2LAB).astype(np.float32)
    profile = np.median(lab, axis=0)                    # สีเฉลี่ยของแต่ละคอลัมน์
    profile = cv2.GaussianBlur(profile.reshape(1, -1, 3), (1, 5), 0).reshape(-1, 3)

    # สีเนื้อตัวต้านทาน = ค่ากลางของโปรไฟล์ (แถบสีกินพื้นที่ไม่ถึงครึ่ง)
    inner = profile[int(w * 0.08): int(w * 0.92)]
    body_color = np.median(inner, axis=0)
    # เน้นความต่างของ "เฉดสี" (a, b) มากกว่าความสว่าง (L) เพราะแถบขาว/เทา/เงิน
    # สว่างใกล้เคียงเนื้อตัวต้านทาน แต่ไม่มีสีเหลืองอมส้มเหมือนเนื้อตัว
    weight = np.array([1.0, 2.0, 2.0])
    dist = np.linalg.norm((profile - body_color) * weight, axis=1)

    # ตัดปลายทั้งสองข้าง (ส่วนโค้งของหัวตัวต้านทาน) ไม่ให้นับเป็นแถบ
    margin = int(w * 0.06)
    dist[:margin] = 0
    dist[-margin:] = 0

    is_band = dist > 24
    bands, start = [], None
    for x, on in enumerate(is_band):
        if on and start is None:
            start = x
        elif not on and start is not None:
            if x - start >= w * 0.025:                  # กว้างพอที่จะเป็นแถบจริง
                bands.append((start, x))
            start = None

    # แถบค่าความคลาดเคลื่อน (แถบสุดท้าย) อยู่ห่างจากแถบอื่นมากที่สุด
    # ถ้าช่องว่างที่กว้างที่สุดอยู่ด้านซ้าย แปลว่าภาพกลับด้าน -> กลับลำดับ
    if len(bands) >= 3:
        gaps = [bands[i + 1][0] - bands[i][1] for i in range(len(bands) - 1)]
        if np.argmax(gaps) == 0:
            bands = [(w - e, w - s) for s, e in reversed(bands)]
            body = cv2.flip(body, 1)
    return bands, body


def band_features(body, band):
    """คืนค่าสีเฉลี่ยของแถบ: H, S, V และ L, a, b"""
    h = body.shape[0]
    s, e = band
    pad = max(1, (e - s) // 4)                          # ใช้เฉพาะส่วนกลางของแถบ
    patch = body[int(h * 0.35): int(h * 0.65), s + pad: e - pad]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    return [*np.median(hsv, axis=0), *np.median(lab, axis=0)]


# ---------------------------------------------------------------------------
# โปรแกรมหลัก
# ---------------------------------------------------------------------------
def load_labels():
    path = os.path.join(IMAGE_DIR, "labels.csv")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8-sig") as f:
        return {row["file"]: row for row in csv.DictReader(f)}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)
    labels = load_labels()
    files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) +
                   glob.glob(os.path.join(IMAGE_DIR, "*.jpeg")) +
                   glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    if not files:
        print(f"ไม่พบภาพในโฟลเดอร์ {IMAGE_DIR}/")
        return

    rows = []
    for path in files:
        name = os.path.basename(path)
        img = cv2.imread(path)
        rect = find_resistor(img)
        if rect is None:
            print(f"{name}: ❌ หาตัวต้านทานไม่เจอ")
            continue

        body, _ = straighten(img, rect)
        bands, body = find_bands(body)
        status = "✅" if len(bands) == 4 else "⚠️"
        print(f"{name}: {status} พบ {len(bands)} แถบ")

        # วาดผลลัพธ์: ภาพต้นฉบับ (กรอบเขียว) + ตัวต้านทานที่หมุนแล้ว (กรอบแต่ละแถบ)
        shown = img.copy()
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.drawContours(shown, [box], 0, (0, 200, 0), 4)
        strip = body.copy()
        for i, (s, e) in enumerate(bands, start=1):
            cv2.rectangle(strip, (s, 2), (e, strip.shape[0] - 3), (0, 255, 0), 2)
            cv2.putText(strip, str(i), (s, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        scale = shown.shape[1] / strip.shape[1]
        strip = cv2.resize(strip, (shown.shape[1], int(strip.shape[0] * scale)))
        cv2.imwrite(os.path.join(OUTPUT_DIR, name), np.vstack([shown, strip]))

        truth = labels.get(name, {})
        for i, band in enumerate(bands, start=1):
            color = truth.get(f"band{i}", "") if len(bands) == 4 else ""
            rows.append([name, i, *[round(float(v), 1) for v in band_features(body, band)], color])

    with open(os.path.join(DATASET_DIR, "bands.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "band", "H", "S", "V", "L", "a", "b", "color"])
        w.writerows(rows)
    print(f"\nบันทึกภาพผลลัพธ์ไว้ที่ {OUTPUT_DIR}/ และข้อมูลสีไว้ที่ {DATASET_DIR}/bands.csv")


if __name__ == "__main__":
    main()
