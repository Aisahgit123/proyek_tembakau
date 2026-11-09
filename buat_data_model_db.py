import os
import pymysql

from fitur_tembakau import CLASS_NAMES, ekstrak_fitur_citra, normalisasi_nama_kelas
from database import get_database_connection, print_mysql_startup_warning


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "train")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


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
    for nama_folder in sorted(os.listdir(DATASET_DIR)):
        folder = os.path.join(DATASET_DIR, nama_folder)
        if not os.path.isdir(folder):
            continue

        kelas = normalisasi_nama_kelas(nama_folder)
        if kelas not in CLASS_NAMES:
            continue
        kelas_ditemukan.add(kelas)

        for nama_file in sorted(os.listdir(folder)):
            if nama_file.lower().endswith(IMAGE_EXTENSIONS):
                path = os.path.join(folder, nama_file)
                source_path = os.path.relpath(path, BASE_DIR).replace(os.sep, "/")
                sampel.append((path, source_path, nama_file, kelas))

    kelas_hilang = sorted(set(CLASS_NAMES) - kelas_ditemukan)
    if kelas_hilang:
        raise ValueError(f"Folder kelas data latih tidak ditemukan: {kelas_hilang}")
    if not sampel:
        raise ValueError(f"Tidak ada gambar data latih di {DATASET_DIR}")
    return sampel


def jalankan_konversi_murni_train():
    sampel = ambil_sampel_train()
    jumlah_disimpan = 0

    conn = get_database_connection()
    try:
        siapkan_tabel(conn)
        with conn.cursor() as cursor:
            for path, source_path, nama_file, kelas in sampel:
                fitur = ekstrak_fitur_citra(path)
                if fitur is None:
                    raise ValueError(f"Gambar tidak dapat dibaca: {path}")

                nilai = (
                    nama_file,
                    kelas,
                    fitur["luas_daun"],
                    fitur["luas_cacat"],
                    fitur["rasio_cacat"],
                    fitur["mean_h"],
                    fitur["mean_s"],
                    fitur["mean_v"],
                    fitur["rekomendasi"],
                    source_path,
                )
                cursor.execute(
                    """
                    SELECT id FROM fitur_data_train
                    WHERE source_path IS NULL
                      AND nama_file = %s
                      AND REPLACE(LOWER(grade_aktual), 'persial', 'parsial') = %s
                    LIMIT 1
                    """,
                    (nama_file, kelas),
                )
                existing = cursor.fetchone()

                if existing:
                    cursor.execute(
                        """
                        UPDATE fitur_data_train
                        SET nama_file=%s, grade_aktual=%s, luas_daun_px=%s,
                            luas_cacat_px=%s, rasio_cacat_persen=%s,
                            rata_rata_hue=%s, rata_rata_sat=%s, rata_rata_val=%s,
                            rekomendasi_mesin=%s, source_path=%s,
                            waktu_ekstraksi=CURRENT_TIMESTAMP
                        WHERE id=%s
                        """,
                        nilai + (existing[0],),
                    )
                else:
                    cursor.execute(
                        """
                        INSERT INTO fitur_data_train (
                            nama_file, grade_aktual, luas_daun_px, luas_cacat_px,
                            rasio_cacat_persen, rata_rata_hue, rata_rata_sat,
                            rata_rata_val, rekomendasi_mesin, source_path
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            nama_file=VALUES(nama_file),
                            grade_aktual=VALUES(grade_aktual),
                            luas_daun_px=VALUES(luas_daun_px),
                            luas_cacat_px=VALUES(luas_cacat_px),
                            rasio_cacat_persen=VALUES(rasio_cacat_persen),
                            rata_rata_hue=VALUES(rata_rata_hue),
                            rata_rata_sat=VALUES(rata_rata_sat),
                            rata_rata_val=VALUES(rata_rata_val),
                            rekomendasi_mesin=VALUES(rekomendasi_mesin),
                            waktu_ekstraksi=CURRENT_TIMESTAMP
                        """,
                        nilai,
                    )
                jumlah_disimpan += 1
    except pymysql.MySQLError as error:
        print_mysql_startup_warning(error)
        raise
    finally:
        conn.close()

    print("Database fitur: MySQL db_tembakau_ambat")
    print("Tabel: fitur_data_train")
    print(f"Sampel latih tersimpan/diperbarui: {jumlah_disimpan}")
    for kelas in CLASS_NAMES:
        banyak = sum(1 for item in sampel if item[3] == kelas)
        print(f"  {kelas}: {banyak}")
    print("Sumber data: dataset/train saja; folder test tidak disentuh.")


if __name__ == "__main__":
    jalankan_konversi_murni_train()
