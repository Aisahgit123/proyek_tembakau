import os
import csv
import datetime
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_KERAS = os.path.join(BASE_DIR, "model", "mobilenet_tembakau.keras")
MODEL_PATH_H5 = os.path.join(BASE_DIR, "model", "mobilenet_tembakau.h5")

# Folder khusus untuk menampung seluruh hasil klasifikasi dan laporan resmi
OUTPUT_DIR = os.path.join(BASE_DIR, "hasil_klasifikasi")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Urutan alfabetis kelas baku
CLASS_NAMES = ["afkir_total", "siap_rajang", "sortir_parsial"]
LABEL_MAP = {
    0: "Afkir Total (Grade C)",
    1: "Siap Rajang (Grade A)",
    2: "Sortir Parsial (Grade B)"
}

def cari_folder_test():
    """Mencari letak folder data uji secara otomatis"""
    kemungkinan_jalur = [
        os.path.join(BASE_DIR, "test"),
        os.path.join(BASE_DIR, "Test"),
        os.path.join(BASE_DIR, "data_uji"),
        os.path.join(BASE_DIR, "dataset", "test"),
        os.path.join(BASE_DIR, "dataset", "Test")
    ]
    for p in kemungkinan_jalur:
        if os.path.exists(p) and os.path.isdir(p):
            sub = [s for s in os.listdir(p) if os.path.isdir(os.path.join(p, s))]
            if any("siap" in s.lower() or "sortir" in s.lower() or "afkir" in s.lower() or "grade" in s.lower() for s in sub):
                return p
    return os.path.join(BASE_DIR, "test")

def muat_model():
    """Memuat bobot model MobileNetV2 dari folder model/"""
    if os.path.exists(MODEL_PATH_KERAS):
        print(f"-> Memuat model format Keras: {MODEL_PATH_KERAS}")
        return tf.keras.models.load_model(MODEL_PATH_KERAS, compile=False)
    elif os.path.exists(MODEL_PATH_H5):
        print(f"-> Memuat model format H5: {MODEL_PATH_H5}")
        return tf.keras.models.load_model(MODEL_PATH_H5, compile=False)
    else:
        raise FileNotFoundError("[ERROR] Model AI tidak ditemukan di folder 'model/'!")

def evaluasi_data_test():
    target_dir = cari_folder_test()

    print("=" * 70)
    print("      MEMULAI PENGUJIAN OTOMATIS DATA TEST (BAB 4 SKRIPSI)")
    print(f"      Folder Sumber Data Uji    : {target_dir}")
    print(f"      Folder Luaran Klasifikasi : {OUTPUT_DIR}")
    print("=" * 70)

    if not os.path.exists(target_dir):
        print(f"[ERROR] Folder pengujian '{target_dir}' tidak ditemukan!")
        return

    model = muat_model()

    subfolders = [s for s in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, s))]
    if not subfolders:
        print("[ERROR] Folder test tidak memiliki sub-folder kelas!")
        return

    y_true = []
    y_pred = []
    rincian_prediksi = []
    daftar_salah = []
    total_diuji = 0

    for sub in sorted(subfolders):
        sub_lower = sub.lower()
        if "afkir" in sub_lower or "grade_c" in sub_lower or sub_lower == "c":
            label_idx = 0
            label_name = "afkir_total"
        elif "siap" in sub_lower or "grade_a" in sub_lower or sub_lower == "a":
            label_idx = 1
            label_name = "siap_rajang"
        elif "sortir" in sub_lower or "grade_b" in sub_lower or sub_lower == "b":
            label_idx = 2
            label_name = "sortir_parsial"
        else:
            continue

        folder_kelas = os.path.join(target_dir, sub)
        files = [f for f in os.listdir(folder_kelas) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"-> Memeriksa Sub-folder: '{sub}' ({len(files)} Citra) | Label Asli: {label_name.upper()}")

        for file_foto in sorted(files):
            path_foto = os.path.join(folder_kelas, file_foto)
            bgr = cv2.imread(path_foto)
            if bgr is None:
                continue

            total_diuji += 1
            img_224 = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_AREA)
            img_rgb = cv2.cvtColor(img_224, cv2.COLOR_BGR2RGB)
            input_tensor = tf.convert_to_tensor(np.expand_dims(img_rgb.astype(np.float32), axis=0))

            pred_tensor = model(input_tensor, training=False)
            probabilitas = pred_tensor.numpy()[0]
            idx_pred = int(np.argmax(probabilitas))
            confidence_score = float(probabilitas[idx_pred]) * 100.0

            y_true.append(label_idx)
            y_pred.append(idx_pred)

            status_tebakan = "BENAR" if label_idx == idx_pred else "SALAH"

            data_baris = {
                "file": file_foto,
                "kelas_asli": LABEL_MAP[label_idx],
                "prediksi_model": LABEL_MAP[idx_pred],
                "confidence": round(confidence_score, 2),
                "prob_afkir": round(float(probabilitas[0]) * 100.0, 2),
                "prob_siap": round(float(probabilitas[1]) * 100.0, 2),
                "prob_sortir": round(float(probabilitas[2]) * 100.0, 2),
                "status": status_tebakan
            }
            rincian_prediksi.append(data_baris)

            if label_idx != idx_pred:
                daftar_salah.append(data_baris)

    if total_diuji == 0:
        print("\n[PERINGATAN] Tidak ada citra yang berhasil diproses di folder test.")
        return

    akurasi = accuracy_score(y_true, y_pred) * 100.0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    target_names = ["Afkir Total (Grade C)", "Siap Rajang (Grade A)", "Sortir Parsial (Grade B)"]
    report_text = classification_report(y_true, y_pred, target_names=target_names, digits=4)

    print("\n" + "=" * 70)
    print("        RINGKASAN HASIL EVALUASI DATA TEST UNTUK BAB 4")
    print("=" * 70)
    print(f"Total Sampel Diuji   : {total_diuji} Citra")
    print(f"Tebakan Benar        : {total_diuji - len(daftar_salah)} Citra")
    print(f"Tebakan Salah        : {len(daftar_salah)} Citra")
    print(f"NILAI AKURASI GLOBAL : {akurasi:.2f}%")
    print("=" * 70)
    print("\nTABEL CONFUSION MATRIX (3 x 3):")
    print("                     | Pred: Afkir (C)  | Pred: Siap (A)   | Pred: Sortir (B)")
    print("-" * 75)
    print(f"Aktual: Afkir (C)    | {cm[0][0]:<16} | {cm[0][1]:<16} | {cm[0][2]:<16}")
    print(f"Aktual: Siap (A)     | {cm[1][0]:<16} | {cm[1][1]:<16} | {cm[1][2]:<16}")
    print(f"Aktual: Sortir (B)   | {cm[2][0]:<16} | {cm[2][1]:<16} | {cm[2][2]:<16}")
    print("-" * 75)
    print("\nRINCIAN METRIK FORMAL PER KELAS (CLASSIFICATION REPORT):")
    print(report_text)

    path_txt = os.path.join(OUTPUT_DIR, "laporan_klasifikasi.txt")
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("LAPORAN RESMI HASIL KLASIFIKASI DATA TEST (BAB 4 SKRIPSI)\n")
        f.write(f"Waktu Pengujian : {datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n")
        f.write(f"Model Evaluasi  : MobileNetV2 Transfer Learning\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total Sampel Uji     : {total_diuji} Citra\n")
        f.write(f"Tebakan Benar        : {total_diuji - len(daftar_salah)} Citra\n")
        f.write(f"Tebakan Salah        : {len(daftar_salah)} Citra\n")
        f.write(f"Akurasi Rata-Rata    : {akurasi:.2f}%\n\n")
        f.write("TABEL CONFUSION MATRIX (3x3):\n")
        f.write(f"[[{cm[0][0]}, {cm[0][1]}, {cm[0][2]}],\n")
        f.write(f" [{cm[1][0]}, {cm[1][1]}, {cm[1][2]}],\n")
        f.write(f" [{cm[2][0]}, {cm[2][1]}, {cm[2][2]}]]\n\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write(report_text + "\n\n")
        f.write("DAFTAR CITRA YANG SALAH KLASIFIKASI:\n")
        for i, item in enumerate(daftar_salah, 1):
            f.write(f"{i}. Berkas: {item['file']} | Asli: {item['kelas_asli']} -> Prediksi: {item['prediksi_model']} ({item['confidence']}%)\n")

    path_csv = os.path.join(OUTPUT_DIR, "rekap_prediksi_per_citra.csv")
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "kelas_asli", "prediksi_model", "confidence",
            "prob_afkir", "prob_siap", "prob_sortir", "status"
        ])
        writer.writeheader()
        writer.writerows(rincian_prediksi)

    label_tampilan = ["Afkir (C)", "Siap Rajang (A)", "Sortir (B)"]
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_tampilan,
        yticklabels=label_tampilan,
        cbar=True,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title(f"Confusion Matrix Pengujian Data Uji\nAkurasi Global: {akurasi:.2f}%", fontsize=12, pad=12, weight="bold")
    plt.xlabel("Predicted Label (Tebakan Model AI)", fontsize=11, labelpad=8)
    plt.ylabel("True Label (Kelas Aktual)", fontsize=11, labelpad=8)
    plt.tight_layout()

    # Simpan grafik visual ke dalam folder hasil_klasifikasi
    path_gambar = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path_gambar, dpi=300)
    plt.close()

    print("\n" + "=" * 70)
    print("[SUKSES] SEMUA BERKAS HASIL KLASIFIKASI LENGKAP TELAH DISIMPAN:")
    print(f"📁 Folder Utama : {OUTPUT_DIR}")
    print(f" 1. 🖼️  Gambar Grafik   : {os.path.basename(path_gambar)} (Heatmap biru 300 DPI)")
    print(f" 2. 📝 Dokumen Laporan : {os.path.basename(path_txt)} (Ringkasan metrik & daftar salah)")
    print(f" 3. 📊 Tabel Excel/CSV : {os.path.basename(path_csv)} (Rincian 30 foto & probabilitas)")
    print("=" * 70)

if __name__ == "__main__":
    evaluasi_data_test()