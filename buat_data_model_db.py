import os
import pymysql

from fitur_tembakau import (
    CLASS_NAMES,
    REKOMENDASI_PER_KELAS,
    ekstrak_fitur_citra,
    normalisasi_nama_kelas,
)
from database import get_database_connection, print_mysql_startup_warning


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "train")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
URUTAN_TAMPILAN_KELAS = ["siap_rajang", "sortir_parsial", "afkir_total"]


def siapkan_tabel(conn):
    with conn.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS fitur_data_train (
                id INT AUTO_INCREMENT PRIMARY KEY,
                nama_file VARCHAR(255) NOT NULL,
                grade_aktual VARCHAR(100) NOT NULL,
                luas_daun_px INT NOT NULL,
                luas_cacat_px INT NOT NULL,
                rasio_cacat_persen FLOAT NOT NULL,
                rata_rata_hue FLOAT NOT NULL,
                rata_rata_sat FLOAT NOT NULL,
                rata_rata_val FLOAT NOT NULL,
                rekomendasi_mesin TEXT NOT NULL,
                waktu_ekstraksi TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source_path VARCHAR(768) NULL,
                UNIQUE KEY idx_fitur_data_train_source_path (source_path)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        cursor.execute("SHOW COLUMNS FROM fitur_data_train")
        columns = {row[0] for row in cursor.fetchall()}
        if "source_path" not in columns:
            cursor.execute(
                "ALTER TABLE fitur_data_train ADD COLUMN source_path VARCHAR(768) NULL"
            )
        cursor.execute("SHOW INDEX FROM fitur_data_train")
        indexes = {row[2] for row in cursor.fetchall()}
        if "idx_fitur_data_train_source_path" not in indexes:
            cursor.execute(
                "CREATE UNIQUE INDEX idx_fitur_data_train_source_path "
                "ON fitur_data_train(source_path)"
            )
        cursor.execute(
            """
            UPDATE fitur_data_train
            SET grade_aktual = REPLACE(LOWER(grade_aktual), 'persial', 'parsial')
            """
        )


def ambil_sampel_train():
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(f"Folder data latih tidak ditemukan: {DATASET_DIR}")

    sampel = []
    kelas_ditemukan = set()
    for nama_folder in os.listdir(DATASET_DIR):
        folder = os.path.join(DATASET_DIR, nama_folder)
        if not os.path.isdir(folder):
            continue

        kelas = normalisasi_nama_kelas(nama_folder)
        if kelas not in CLASS_NAMES:
            continue
        kelas_ditemukan.add(kelas)

        for nama_file in sorted(os.listdir(folder), key=str.casefold):
            if nama_file.lower().endswith(IMAGE_EXTENSIONS):
                path = os.path.join(folder, nama_file)
                source_path = os.path.relpath(path, BASE_DIR).replace(os.sep, "/")
                sampel.append((path, source_path, nama_file, kelas))

    urutan_kelas = {kelas: index for index, kelas in enumerate(URUTAN_TAMPILAN_KELAS)}
    sampel.sort(key=lambda item: (urutan_kelas[item[3]], item[2].casefold()))

    kelas_hilang = sorted(set(CLASS_NAMES) - kelas_ditemukan)
    if kelas_hilang:
        raise ValueError(f"Folder kelas data latih tidak ditemukan: {kelas_hilang}")
    if not sampel:
        raise ValueError(f"Tidak ada gambar data latih di {DATASET_DIR}")
    return sampel


def jalankan_konversi_murni_train():
    sampel = ambil_sampel_train()
    data_fitur = []
    for path, source_path, nama_file, kelas in sampel:
        fitur = ekstrak_fitur_citra(path)
        if fitur is None:
            raise ValueError(f"Gambar tidak dapat dibaca: {path}")
        data_fitur.append(
            (
                nama_file,
                kelas,
                fitur["luas_daun"],
                fitur["luas_cacat"],
                fitur["rasio_cacat"],
                fitur["mean_h"],
                fitur["mean_s"],
                fitur["mean_v"],
                REKOMENDASI_PER_KELAS[kelas],
                source_path,
            )
        )

    conn = get_database_connection()
    try:
        siapkan_tabel(conn)
        conn.begin()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*) FROM fitur_data_train
                WHERE source_path IS NULL OR source_path NOT LIKE %s
                """,
                ("dataset/train/%",),
            )
            if cursor.fetchone()[0]:
                raise ValueError(
                    "Tabel fitur_data_train berisi baris di luar dataset/train; "
                    "penomoran ulang dibatalkan agar data lain tidak berubah."
                )
            cursor.execute("DELETE FROM fitur_data_train")
            cursor.executemany(
                """
                INSERT INTO fitur_data_train (
                    id, nama_file, grade_aktual, luas_daun_px, luas_cacat_px,
                    rasio_cacat_persen, rata_rata_hue, rata_rata_sat,
                    rata_rata_val, rekomendasi_mesin, source_path
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    (row_id, *nilai)
                    for row_id, nilai in enumerate(data_fitur, start=1)
                ),
            )
        conn.commit()
        with conn.cursor() as cursor:
            cursor.execute(
                f"ALTER TABLE fitur_data_train AUTO_INCREMENT = {len(data_fitur) + 1}"
            )
    except pymysql.MySQLError as error:
        conn.rollback()
        print_mysql_startup_warning(error)
        raise
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print("Database fitur: MySQL db_tembakau_ambat")
    print("Tabel: fitur_data_train")
    print(f"Sampel latih tersimpan/diperbarui: {len(data_fitur)}")
    for kelas in URUTAN_TAMPILAN_KELAS:
        banyak = sum(1 for item in sampel if item[3] == kelas)
        print(f"  {kelas}: {banyak}")
    print("Sumber data: dataset/train saja; folder test tidak disentuh.")


if __name__ == "__main__":
    jalankan_konversi_murni_train()
