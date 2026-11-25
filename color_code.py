"""
color_code.py — แปลงค่าความต้านทาน (Ω) เป็นแถบสี 4 แถบ และวาดภาพตัวต้านทาน

ใช้ใน Terminal:  python color_code.py 4700
"""
import math
import sys

import cv2
import numpy as np

DIGIT_COLORS = ["black", "brown", "red", "orange", "yellow",
                "green", "blue", "violet", "grey", "white"]
MULT_COLORS = {-2: "silver", -1: "gold", **{i: c for i, c in enumerate(DIGIT_COLORS)}}
TOLERANCE_COLORS = {5: "gold", 10: "silver", 1: "brown", 2: "red"}
THAI = {"black": "ดำ", "brown": "น้ำตาล", "red": "แดง", "orange": "ส้ม",
        "yellow": "เหลือง", "green": "เขียว", "blue": "น้ำเงิน", "violet": "ม่วง",
        "grey": "เทา", "white": "ขาว", "gold": "ทอง", "silver": "เงิน"}
BGR = {"black": (30, 30, 30), "brown": (20, 60, 120), "red": (40, 40, 210),
       "orange": (20, 125, 245), "yellow": (40, 215, 245), "green": (60, 150, 40),
       "blue": (190, 95, 25), "violet": (160, 50, 130), "grey": (135, 135, 135),
       "white": (245, 245, 245), "gold": (55, 165, 205), "silver": (195, 195, 195)}

MIN_OHMS, MAX_OHMS = 0.1, 99e9      # ช่วงที่ตัวต้านทาน 4 แถบแสดงได้


def encode(ohms, tolerance=5):
    """แปลงค่าโอห์มเป็นสี 4 แถบ
    คืนค่า (colors, ค่าที่แสดงได้จริง) — ตัวต้านทาน 4 แถบมีเลขนัยสำคัญได้แค่ 2 หลัก
    ถ้าค่าที่ใส่มีมากกว่า 2 หลัก จะถูกปัดเป็น 2 หลัก
    """
    if not MIN_OHMS <= ohms <= MAX_OHMS:
        raise ValueError(f"ค่าต้องอยู่ระหว่าง {MIN_OHMS} Ω ถึง 99 GΩ")
    exp = math.floor(math.log10(ohms)) - 1          # เลขชี้กำลังของตัวคูณ
    two_digits = round(ohms / 10 ** exp)
    if two_digits == 100:                           # เช่น 99.6 ปัดเป็น 100
        two_digits, exp = 10, exp + 1
    if exp < -2:                                    # ค่าน้อยกว่า 1 Ω มาก
        raise ValueError("ค่าน้อยเกินกว่าที่ตัวต้านทาน 4 แถบแสดงได้")
    colors = [DIGIT_COLORS[two_digits // 10], DIGIT_COLORS[two_digits % 10],
              MULT_COLORS[exp], TOLERANCE_COLORS[tolerance]]
    actual = round(two_digits * 10.0 ** exp, 2)
    return colors, actual


def draw_resistor(colors, width=900, height=260):
    """วาดภาพตัวต้านทาน (BGR) ตามสีที่กำหนด"""
    img = np.full((height, width, 3), 255, np.uint8)
    cy = height // 2
    cv2.line(img, (0, cy), (width, cy), (170, 172, 176), 10, cv2.LINE_AA)

    x0, x1, r, mid = int(width * 0.2), int(width * 0.8), int(height * 0.26), int(height * 0.19)
    body = np.zeros((height, width), np.uint8)
    cv2.circle(body, (x0 + r, cy), r, 255, -1, cv2.LINE_AA)
    cv2.circle(body, (x1 - r, cy), r, 255, -1, cv2.LINE_AA)
    cv2.rectangle(body, (x0 + r, cy - mid), (x1 - r, cy + mid), 255, -1)
    img[body > 0] = (150, 200, 228)

    span = x1 - x0
    positions = [x0 + span * 0.22, x0 + span * 0.34, x0 + span * 0.46, x0 + span * 0.78]
    bw = int(span * 0.06)
    for name, bx in zip(colors, positions):
        stripe = np.zeros_like(body)
        cv2.rectangle(stripe, (int(bx) - bw // 2, 0), (int(bx) + bw // 2, height), 255, -1)
        img[(stripe > 0) & (body > 0)] = BGR[name]

    # เงาให้ดูกลม: มืดลงที่ขอบบน-ล่างของตัว
    rows = np.abs(np.arange(height) - (cy - mid * 0.25)) / (r * 1.1)
    shade = np.clip(1 - 0.35 * rows ** 2, 0.55, 1).reshape(-1, 1, 1)
    shaded = (img * shade).astype(np.uint8)
    img[body > 0] = shaded[body > 0]
    edge = cv2.morphologyEx(body, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    img[edge > 0] = (90, 110, 125)
    return img


def format_ohms(value):
    for limit, unit in ((1e9, "GΩ"), (1e6, "MΩ"), (1e3, "kΩ")):
        if value >= limit:
            return f"{value / limit:g} {unit}"
    return f"{value:g} Ω"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("วิธีใช้: python color_code.py <ค่าโอห์ม>   เช่น  python color_code.py 4700")
        sys.exit()
    colors, actual = encode(float(sys.argv[1]))
    print(f"ค่า {format_ohms(actual)} ±5%")
    print("แถบสี: " + " ".join(THAI[c] for c in colors))
    cv2.imwrite("color_code.png", draw_resistor(colors))
    print("บันทึกภาพไว้ที่ color_code.png")
