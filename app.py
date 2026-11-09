import os
import datetime
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
import pymysql

from database import get_database_connection, print_mysql_startup_warning
from vision import (
    hitung_persentase_cokelat,
    make_transparent_leaf_preview,
    prepare_leaf_sample,
)

app = Flask(__name__, template_folder="templates", static_folder="static")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
SEGMENTED_DIR = os.path.join(BASE_DIR, "static", "segmented")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SEGMENTED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Urutan alfabetis kelas baku
CLASS_NAMES = ["afkir_total", "siap_rajang", "sortir_parsial"]
MODEL = None

def muat_model_ai():
    """
    Memuat model MobileNetV2 dalam format Keras modern.
    """
    global MODEL
    if MODEL is not None:
        return MODEL

    # Kumpulkan semua kemungkinan lokasi file model di laptop Aisah
    daftar_kemungkinan = [
        os.path.join(MODEL_DIR, "mobilenet_tembakau.keras"),
        os.path.join(BASE_DIR, "mobilenet_tembakau.keras"),
    ]

    print("=" * 65)
    print("-> [PROSES] Mencari dan memuat model AI ke memori RAM...")
    model_terpilih = None

    for jalur in daftar_kemungkinan:
        if os.path.exists(jalur) and os.path.getsize(jalur) > 1000:
            try:
                print(f"-> Ditemukan berkas model di: {jalur} ({os.path.getsize(jalur)/(1024*1024):.2f} MB)")
                # compile=False mencegah error konfigurasi optimizer/loss
                model_terpilih = tf.keras.models.load_model(jalur, compile=False)
                print(f"-> [SUKSES BESAR] Berhasil memuat model dari: {os.path.basename(jalur)}")
                break
            except Exception as e:
                print(f"-> [PERINGATAN] Gagal memuat dari {jalur}: {e}")

    if model_terpilih is not None:
        try:
            # Lakukan pemanasan (warm-up) 1 kali agar graph tensor siap di memori RAM
            dummy_input = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
            _ = model_terpilih(dummy_input, training=False)
            print("-> [STATUS] Warm-up model sukses, model AI 100% SIAP DI MEMORI!")
        except Exception as e_warm:
            print(f"-> [INFO] Warm-up bypass: {e_warm}")
        
        MODEL = model_terpilih
        print("=" * 65)
        return MODEL

    print("[ERROR FATAL] Berkas model AI (.keras) tidak ditemukan atau gagal dimuat!")
    print(f"Jalur yang diperiksa: {daftar_kemungkinan}")
    print("=" * 65)
    return None

# Panggil fungsi saat file app.py pertama kali dijalankan agar langsung masuk RAM
muat_model_ai()

def init_database():
    conn = get_database_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inspection_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    grade VARCHAR(10) NOT NULL,
                    confidence FLOAT NOT NULL,
                    defect_percentage FLOAT NOT NULL,
                    recommendation TEXT NOT NULL,
                    action_step TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
        print("-> [SUKSES] Terhubung ke MySQL XAMPP; tabel inspection_history siap.")
    finally:
        conn.close()

try:
    init_database()
except pymysql.MySQLError as error:
    print_mysql_startup_warning(error)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global MODEL
    if MODEL is None:
        MODEL = muat_model_ai()
        if MODEL is None:
            return jsonify({
                "status": "error",
                "message": "Model AI belum siap di memori! Pastikan berkas 'mobilenet_tembakau.keras' ada di folder 'model/'."
            }), 500

    try:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        citra_bgr = None
        nama_bersih = f"daun_{timestamp_str}.jpg"
        path_simpan_asli = os.path.join(UPLOAD_DIR, nama_bersih)

        if request.files:
            file_key = list(request.files.keys())[0]
            file = request.files[file_key]
            if file and file.filename != "":
                nama_file_asli = secure_filename(file.filename) or "upload.jpg"
                nama_bersih = f"daun_{timestamp_str}_{nama_file_asli}"
                path_simpan_asli = os.path.join(UPLOAD_DIR, nama_bersih)
                file.save(path_simpan_asli)
                citra_bgr = cv2.imread(path_simpan_asli)

        if citra_bgr is None:
            return jsonify({"status": "error", "message": "Foto daun tidak berhasil dibaca."}), 400

        citra_segmentasi, mask_daun, persentase_cacat, citra_rgb = (
            prepare_leaf_sample(citra_bgr)
        )
        persentase_cokelat = hitung_persentase_cokelat(
            citra_segmentasi, mask_daun
        )

        nama_seg = f"seg_{os.path.splitext(nama_bersih)[0]}.png"
        path_simpan_seg = os.path.join(SEGMENTED_DIR, nama_seg)
        mask_pratinjau = cv2.resize(
            mask_daun,
            (citra_bgr.shape[1], citra_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        hasil_segmentasi = make_transparent_leaf_preview(citra_bgr, mask_pratinjau)
        if not cv2.imwrite(path_simpan_seg, hasil_segmentasi):
            raise OSError(f"Gagal menyimpan hasil segmentasi: {path_simpan_seg}")

        input_tensor = tf.convert_to_tensor(np.expand_dims(citra_rgb, axis=0))
        
        prediksi_output = MODEL(input_tensor, training=False)
        prediksi_prob = prediksi_output.numpy()[0]

        prob_afkir = float(prediksi_prob[0]) * 100.0
        prob_siap = float(prediksi_prob[1]) * 100.0
        prob_sortir = float(prediksi_prob[2]) * 100.0
        prob_grade = {
            "A": prob_siap,
            "B": prob_sortir,
            "C": prob_afkir,
        }

        idx_tertinggi = int(np.argmax(prediksi_prob))
        kelas_model = CLASS_NAMES[idx_tertinggi]
        kelas_terpilih = kelas_model
        if kelas_model == "afkir_total" and persentase_cokelat <= 25.0:
            kelas_terpilih = "sortir_parsial"
        elif kelas_model == "siap_rajang" and persentase_cokelat >= 5.0:
            kelas_terpilih = "sortir_parsial"
        idx_keputusan = CLASS_NAMES.index(kelas_terpilih)
        confidence = round(float(prediksi_prob[idx_keputusan]) * 100.0, 1)
        dasar_keputusan = (
            f"Model memilih afkir dan area cokelat {persentase_cokelat}% "
            "melebihi batas 25%."
            if kelas_model == "afkir_total" and kelas_terpilih == "afkir_total"
            else (
                f"Model memilih afkir, tetapi area cokelat hanya "
                f"{persentase_cokelat}%; daun masuk Grade B."
                if kelas_model == "afkir_total"
                else f"Area cokelat daun {persentase_cokelat}%."
            )
        )

        keputusan_kelas = {
            "siap_rajang": (
                "A",
                "Sangat Layak (Siap Rajang Penuh)",
                f"Daun memenuhi klasifikasi siap rajang. {dasar_keputusan} "
                f"Cacat tersegmentasi: {persentase_cacat}%.",
                "Langsung masukkan daun ke corong mesin perajang.",
                "Jaga kelembapan daun sebelum perajangan.",
            ),
            "sortir_parsial": (
                "B",
                "Perlu Sortir Parsial (Rompes Ujung)",
                f"Daun diklasifikasikan sebagai sortir parsial. {dasar_keputusan} "
                f"Cacat tersegmentasi: {persentase_cacat}%.",
                "Potong bagian ujung daun yang cacat sebelum dimasukkan ke corong mesin.",
                "Bagian helai yang bersih tetap dapat dirajang.",
            ),
            "afkir_total": (
                "C",
                "Afkir Total (Ditolak Mesin)",
                f"Daun diklasifikasikan sebagai afkir total. {dasar_keputusan} "
                f"Cacat tersegmentasi: {persentase_cacat}%.",
                "Jangan masukkan daun ke corong mesin perajang.",
                "Singkirkan daun dari tumpukan pemeraman.",
            ),
        }
        huruf, judul, deskripsi, tindakan, penanganan = keputusan_kelas[
            kelas_terpilih
        ]
        grade = f"GRADE {huruf}"

        try:
            conn = get_database_connection()
            try:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO inspection_history
                            (filename, grade, confidence, defect_percentage, recommendation, action_step)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (nama_bersih, huruf, confidence, persentase_cacat, tindakan, penanganan))
            finally:
                conn.close()
        except pymysql.MySQLError as e_db:
            print_mysql_startup_warning(e_db)
            return jsonify({
                "status": "error",
                "message": "Hasil inspeksi tidak dapat disimpan. Pastikan MySQL XAMPP aktif.",
            }), 503

        return jsonify({
            "status": "success",
            "data": {
                "grade": grade,
                "huruf": huruf,
                "judul": judul,
                "deskripsi": deskripsi,
                "confidence": confidence,
                "persentase_cacat": persentase_cacat,
                "persentase_cokelat": persentase_cokelat,
                "probabilitas": {
                    "grade_a": round(prob_grade["A"], 1),
                    "grade_b": round(prob_grade["B"], 1),
                    "grade_c": round(prob_grade["C"], 1)
                },
                "rekomendasi": {
                    "tindakan": tindakan,
                    "penanganan": penanganan
                },
                "url_asli": f"/static/uploads/{nama_bersih}",
                "url_mask": f"/static/segmented/{nama_seg}"
            }
        })

    except ValueError as e_pred:
        return jsonify({"status": "error", "message": str(e_pred)}), 400
    except Exception as e_pred:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e_pred)}), 500

@app.route("/history", methods=["GET"])
def get_history():
    try:
        conn = get_database_connection()
        try:
            with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute("SELECT * FROM inspection_history ORDER BY id DESC LIMIT 50")
                rows = cursor.fetchall()
        finally:
            conn.close()

        hasil = []
        for r in rows:
            hasil.append({
                "id": r["id"],
                "filename": r["filename"],
                "grade": r["grade"],
                "confidence": r["confidence"],
                "defect_percentage": r["defect_percentage"],
                "recommendation": r["recommendation"],
                "created_at": r["created_at"]
            })
        return jsonify({"status": "success", "history": hasil})
    except pymysql.MySQLError as error:
        print_mysql_startup_warning(error)
        return jsonify({
            "status": "error",
            "message": "Riwayat tidak tersedia. Pastikan MySQL XAMPP aktif.",
        }), 503
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/clear-history", methods=["POST"])
def clear_history():
    try:
        conn = get_database_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("TRUNCATE TABLE inspection_history")
        finally:
            conn.close()
        return jsonify({"status": "success", "message": "Riwayat berhasil dikosongkan"})
    except pymysql.MySQLError as error:
        print_mysql_startup_warning(error)
        return jsonify({
            "status": "error",
            "message": "Riwayat tidak dapat dikosongkan. Pastikan MySQL XAMPP aktif.",
        }), 503
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    print("=" * 65)
    print("SERVER TOBACCOVISION (DESA AMBAT, PAMEKASAN) TELAH AKTIF!")
    print("Silakan buka di browser laptop: http://127.0.0.1:5000")
    print("=" * 65)
    app.run(host="0.0.0.0", port=5000, debug=False)