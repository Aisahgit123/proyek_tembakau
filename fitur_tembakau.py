import os

import cv2
import numpy as np


CLASS_NAMES = ["afkir_total", "siap_rajang", "sortir_parsial"]
REKOMENDASI_PER_KELAS = {
    "siap_rajang": "Grade A: Siap Rajang Penuh (Langsung ke corong)",
    "sortir_parsial": "Grade B: Sortir Parsial (Rompes ujung daun)",
    "afkir_total": "Grade C: Afkir Total (Ditolak dari mesin)",
}
FEATURE_COLUMNS = [
    "luas_daun_px",
    "luas_cacat_px",
    "rasio_cacat_persen",
    "rata_rata_hue",
    "rata_rata_sat",
    "rata_rata_val",
]


def normalisasi_nama_kelas(nama):
    return nama.casefold().replace("persial", "parsial")


def ekstrak_fitur_citra(citra_atau_path):
    if isinstance(citra_atau_path, (str, os.PathLike)):
        citra_bgr = cv2.imread(os.fspath(citra_atau_path))
    else:
        citra_bgr = citra_atau_path
    if citra_bgr is None:
        return None

    citra_224 = cv2.resize(citra_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(citra_224, cv2.COLOR_BGR2HSV)

    lower_leaf = np.array([10, 36, 35])
    upper_leaf = np.array([95, 255, 255])
    mask_leaf = cv2.inRange(hsv, lower_leaf, upper_leaf)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_leaf = cv2.morphologyEx(mask_leaf, cv2.MORPH_CLOSE, kernel)

    total_piksel_daun = int(cv2.countNonZero(mask_leaf))
    if total_piksel_daun == 0:
        total_piksel_daun = 224 * 224

    lower_defect = np.array([0, 25, 20])
    upper_defect = np.array([25, 255, 120])
    mask_defect = cv2.inRange(hsv, lower_defect, upper_defect)
    mask_defect_in_leaf = cv2.bitwise_and(mask_defect, mask_defect, mask=mask_leaf)
    total_piksel_cacat = int(cv2.countNonZero(mask_defect_in_leaf))
    rasio_cacat = (total_piksel_cacat / total_piksel_daun) * 100.0

    piksel_daun = mask_leaf > 0
    h_channel = hsv[:, :, 0][piksel_daun]
    s_channel = hsv[:, :, 1][piksel_daun]
    v_channel = hsv[:, :, 2][piksel_daun]

    mean_h = float(np.mean(h_channel)) if len(h_channel) > 0 else 0.0
    mean_s = float(np.mean(s_channel)) if len(s_channel) > 0 else 0.0
    mean_v = float(np.mean(v_channel)) if len(v_channel) > 0 else 0.0

    if rasio_cacat < 5.0:
        rekomendasi = "Grade A: Siap Rajang Penuh (Langsung ke corong)"
    elif rasio_cacat <= 25.0:
        rekomendasi = "Grade B: Sortir Parsial (Rompes ujung daun)"
    else:
        rekomendasi = "Grade C: Afkir Total (Ditolak dari mesin)"

    return {
        "luas_daun": total_piksel_daun,
        "luas_cacat": total_piksel_cacat,
        "rasio_cacat": round(rasio_cacat, 2),
        "mean_h": round(mean_h, 2),
        "mean_s": round(mean_s, 2),
        "mean_v": round(mean_v, 2),
        "rekomendasi": rekomendasi,
    }
