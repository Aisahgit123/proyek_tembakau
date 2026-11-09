import os
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_KERAS = os.path.join(BASE_DIR, "model", "mobilenet_tembakau.keras")
MODEL_H5 = os.path.join(BASE_DIR, "model", "mobilenet_tembakau.h5")

def bedah_model():
    path_model = None
    if os.path.exists(MODEL_KERAS):
        path_model = MODEL_KERAS
    elif os.path.exists(MODEL_H5):
        path_model = MODEL_H5
    else:
        print("[ERROR] Berkas model tidak ditemukan di folder 'model/'!")
        return

    ukuran_mb = os.path.getsize(path_model) / (1024 * 1024)
    print("=" * 75)
    print("         BEDAH STRUKTUR MATRIKS DAN BOBOT MODEL MOBILENETV2")
    print(f"Berkas Model   : {os.path.basename(path_model)}")
    print(f"Ukuran Berkas  : {ukuran_mb:.2f} MB")
    print("=" * 75)

    model = tf.keras.models.load_model(path_model, compile=False)

    total_bobot_angka = 0
    rincian_lapisan = []

    print("\n--- RINGKASAN STRUKTUR RAK / LAPISAN DI DALAM MODEL ---")
    print(f"{'No':<4} | {'Nama Lapisan (Layer)':<32} | {'Bentuk Tensor (Shape)':<22} | {'Jumlah Bobot'}")
    print("-" * 75)

    idx = 1
    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights) > 0:
            total_layer_params = sum(w.size for w in weights)
            total_bobot_angka += total_layer_params
            shape_str = str(weights[0].shape)
            print(f"{idx:<4} | {layer.name:<32} | {shape_str:<22} | {total_layer_params:,} angka")
            rincian_lapisan.append((layer.name, weights))
            idx += 1

    print("-" * 75)
    print(f"TOTAL PARAMETER BOBOT NUMERIK: {total_bobot_angka:,} ANGKA DESIMAL")
    print("=" * 75)

    print("\n--- CONTOH WUJUD ASLI ANGKA-ANGKA PINTAR (SAMPLE WEIGHTS) ---")
    
    lapisan_awal = rincian_lapisan[0]
    sampel_awal = lapisan_awal[1][0].flatten()[:8]
    print(f"\n1. Lapisan Ekstraksi Tekstur ({lapisan_awal[0]}):")
    print("   Contoh 8 angka bobot pertama:")
    print("   ", [round(float(val), 6) for val in sampel_awal])

    lapisan_dense = rincian_lapisan[-1]
    sampel_dense = lapisan_dense[1][0].flatten()[:8]
    print(f"\n2. Lapisan Pengambil Keputusan Klasifikasi ({lapisan_dense[0]}):")
    print("   Contoh 8 angka bobot penentu kelas:")
    print("   ", [round(float(val), 6) for val in sampel_dense])

    output_txt = os.path.join(BASE_DIR, "hasil_klasifikasi", "struktur_bobot_model.txt")
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("INSPEKSI ARSITEKTUR BOBOT MODEL MOBILENETV2 TEMBAKAU DESA AMBAT\n")
        f.write("=" * 65 + "\n")
        f.write(f"Ukuran Berkas Fisik : {ukuran_mb:.2f} MB\n")
        f.write(f"Total Bobot Angka   : {total_bobot_angka:,} parameter numerik\n")
        f.write(f"Lapisan Terakhir    : 3 unit output (Grade C, Grade A, Grade B)\n")
        f.write("=" * 65 + "\n\n")
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    print(f"\n[SUKSES] Ringkasan struktur lengkap tersimpan di: {output_txt}")

if __name__ == "__main__":
    bedah_model()