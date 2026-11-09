# Database MySQL XAMPP dan migrasi data lama

Semua operasi database aplikasi berjalan pada MySQL XAMPP. Modul
`database.py` memakai koneksi `localhost:3306` sebagai `root` tanpa kata sandi,
memilih charset `utf8mb4`, mengaktifkan autocommit, dan membuat database
`db_tembakau_ambat` bila belum ada. `app.py` membuat tabel `inspection_history`
saat mulai. Skrip `buat_data_model_db.py` menulis tabel fitur ke database yang
sama.

Skrip satu-kali `migrate_sqlite_to_mysql.py` tersedia hanya untuk menyalin data
lama secara baca-saja. Skrip memindahkan tabel `fitur_data_train` dari
`database/tobacco_features.db` dan, jika tersedia, `inspection_history` dari
`database/tobacco_history.db` ke database `db_tembakau_ambat`. Skrip menyalin
seluruh kolom termasuk ID, mempertahankan indeks dan nilai default yang
didukung, lalu membandingkan jumlah baris sumber dan tujuan sebelum menyatakan
migrasi berhasil. File SQLite dibuka hanya-baca dan tidak diubah.
Jika data SQLite lama perlu dipertahankan, jalankan skrip migrasi sebelum
pertama kali menjalankan `app.py`; aplikasi akan membuat `inspection_history`
secara otomatis dan skrip migrasi tidak menimpa tabel tujuan yang sudah ada.

## Menjalankan migrasi

1. Buka XAMPP Control Panel, lalu jalankan **MySQL**. Aplikasi web tetap dapat
   dimulai jika MySQL belum aktif, tetapi operasi database akan mengembalikan
   error yang meminta layanan MySQL dinyalakan.
2. Buka terminal di direktori proyek dan pasang dependensi:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

   Jika tidak memakai virtual environment proyek, jalankan
   `python -m pip install -r requirements.txt`.
3. Jalankan skrip dari direktori proyek:

   ```powershell
   .\.venv\Scripts\python.exe .\migrate_sqlite_to_mysql.py
   ```

   Untuk instalasi Python global, gunakan `python .\migrate_sqlite_to_mysql.py`.
   Konfigurasi koneksi bawaan adalah `localhost:3306`, pengguna `root`, tanpa
   kata sandi. Skrip membuat database tujuan secara otomatis.
4. Skrip menolak menimpa tabel tujuan yang sudah ada. Jika skrip pernah berjalan
   sebagian atau tabel sudah tersedia, periksa isi database terlebih dahulu.
   Jangan hapus tabel sebelum memastikan data tujuan boleh diganti.

Jika tabel `inspection_history` atau file database riwayat belum ada, skrip
melewati tabel riwayat dan tetap memigrasikan tabel fitur. Database fitur dan
tabel `fitur_data_train` wajib tersedia.

## Verifikasi melalui phpMyAdmin

1. Buka <http://localhost/phpmyadmin>.
2. Pilih database **db_tembakau_ambat** di panel kiri.
3. Pastikan tabel `fitur_data_train` dan, jika sumber tersedia,
   `inspection_history` terlihat.
4. Buka tab **SQL**, jalankan:

   ```sql
   SELECT COUNT(*) AS jumlah_fitur FROM fitur_data_train;
   SELECT COUNT(*) AS jumlah_riwayat FROM inspection_history;
   ```

   Hasil sumber saat ini adalah 90 baris fitur dan 6 baris riwayat. Jumlah
   dapat berbeda jika file SQLite berubah sebelum migrasi. Skrip juga mencetak
   jumlah baris aktual yang sudah dicocokkan.
5. Periksa sampel lengkap dengan membuka tabel lalu memilih **Browse/Jelajahi**,
   atau jalankan `SELECT * FROM fitur_data_train;` dan
   `SELECT * FROM inspection_history;`.

Jika file atau tabel riwayat memang tidak ada, query terhadap
`inspection_history` tidak berlaku. Jangan jalankan skrip berulang kali pada
tabel tujuan yang sudah ada: skrip berhenti agar tidak menggandakan atau
menimpa data.
