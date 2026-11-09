import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from vision import MODEL_IMAGE_SIZE, prepare_leaf_sample

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Urutan alfabetis kelas baku
CLASS_NAMES = ["afkir_total", "siap_rajang", "sortir_parsial"]

def cari_folder_train():
    kemungkinan = [
        os.path.join(BASE_DIR, "dataset", "train"),
        os.path.join(BASE_DIR, "dataset", "Train"),
        os.path.join(BASE_DIR, "train"),
        os.path.join(BASE_DIR, "Train"),
        os.path.join(BASE_DIR, "dataset")
    ]
    for p in kemungkinan:
        if os.path.exists(p) and os.path.isdir(p):
            sub = [s for s in os.listdir(p) if os.path.isdir(os.path.join(p, s))]
            if any("siap" in s.lower() or "sortir" in s.lower() or "afkir" in s.lower() for s in sub):
                return p
    return os.path.join(BASE_DIR, "dataset")

def muat_dataset_dari_folder(target_dir):
    subfolders = [s for s in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, s))]
    subfolders = [s for s in subfolders if s.lower() not in ['test', 'val', 'validation']]

    x_data = []
    y_data = []
    rincian = {}
    jumlah_gagal_segmentasi = 0

    for sub in sorted(subfolders):
        sub_lower = sub.lower()
        if "afkir" in sub_lower or "grade_c" in sub_lower:
            label_idx = 0
            label_name = "afkir_total"
        elif "siap" in sub_lower or "grade_a" in sub_lower:
            label_idx = 1
            label_name = "siap_rajang"
        elif "sortir" in sub_lower or "grade_b" in sub_lower:
            label_idx = 2
            label_name = "sortir_parsial"
        else:
            continue

        folder_kelas = os.path.join(target_dir, sub)
        files = [f for f in os.listdir(folder_kelas) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"-> Membaca Kelas '{sub}' ({label_name}): {len(files)} citra")
        rincian[label_name] = 0

        for f in files:
            p_img = os.path.join(folder_kelas, f)
            bgr = cv2.imread(p_img)
            if bgr is None:
                jumlah_gagal_segmentasi += 1
                continue

            try:
                _, _, _, model_image = prepare_leaf_sample(bgr)
            except ValueError:
                jumlah_gagal_segmentasi += 1
                continue
            x_data.append(model_image)
            y_data.append(label_idx)
            rincian[label_name] += 1

    if jumlah_gagal_segmentasi:
        print(f"-> [PERINGATAN] {jumlah_gagal_segmentasi} citra dilewati karena gagal dibaca/segmentasi.")
    x_arr = np.array(x_data, dtype=np.float32)
    y_arr = np.array(y_data, dtype=np.int32)
    return x_arr, y_arr, rincian


def muat_dataset_train():
    target_dir = cari_folder_train()
    print("=" * 65)
    print("MEMUAT CITRA DAUN TERSEGMENTASI UNTUK PELATIHAN MOBILENETV2")
    print(f"Sumber Folder Train : {target_dir}")
    print("=" * 65)
    return muat_dataset_dari_folder(target_dir)


def bangun_model_mobilenet():
    print("\n-> Memuat base model MobileNetV2 dengan bobot pra-latih ImageNet...")
    
    # Preprocessing bawaan MobileNetV2 (skala [-1, 1])
    input_layer = tf.keras.layers.Input(
        shape=(MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE, 3)
    )
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(input_layer)
    x = tf.keras.layers.RandomRotation(0.08)(x)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    # Bekukan bobot dasar agar stabil
    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output_layer = tf.keras.layers.Dense(3, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def jalankan_pelatihan():
    tf.keras.utils.set_random_seed(42)
    x_train, y_train, rincian = muat_dataset_train()
    total_data = len(x_train)

    if total_data < 15 or len(np.unique(y_train)) != len(CLASS_NAMES):
        raise ValueError(
            "Data tersegmentasi terlalu sedikit atau tidak mencakup semua kelas; "
            "periksa dataset/train dan hasil segmentasi."
        )

    print(f"\n[INFO] Total data siap latih: {total_data} sampel.")
    print(f"Rincian per kelas: {rincian}")
    x_fit, x_validation, y_fit, y_validation = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    model = bangun_model_mobilenet()

    print("\n" + "=" * 65)
    print("MEMULAI PELATIHAN MODEL DENGAN VALIDASI TERSTRATIFIKASI...")
    print("=" * 65)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=3,
            factor=0.3,
            min_lr=1e-5,
        ),
    ]
    history = model.fit(
        x_fit,
        y_fit,
        validation_data=(x_validation, y_validation),
        epochs=40,
        batch_size=8,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )

    base_model = next(
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.Model)
        and "mobilenetv2" in layer.name.lower()
    )
    base_model.trainable = True
    fine_tune_start = len(base_model.layers) - 30
    for index, layer in enumerate(base_model.layers):
        layer.trainable = (
            index >= fine_tune_start
            and not isinstance(layer, tf.keras.layers.BatchNormalization)
        )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("-> Fine-tuning 30 layer terakhir MobileNetV2 dengan learning rate kecil.")
    history = model.fit(
        x_fit,
        y_fit,
        validation_data=(x_validation, y_validation),
        epochs=25,
        batch_size=8,
        shuffle=True,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=5,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                patience=2,
                factor=0.3,
                min_lr=1e-7,
            ),
        ],
        verbose=1,
    )

    test_dir = os.path.join(os.path.dirname(cari_folder_train()), "test")
    if os.path.isdir(test_dir):
        x_test, y_test, _ = muat_dataset_dari_folder(test_dir)
        if len(y_test):
            test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
            print(
                f"-> Evaluasi data test terpisah: "
                f"akurasi {test_accuracy * 100:.1f}% (loss {test_loss:.4f})"
            )
            test_predictions = model.predict(x_test, batch_size=8, verbose=0).argmax(axis=1)
            print(
                "-> Confusion matrix test (baris/kolom: afkir, siap rajang, sortir):\n"
                f"{confusion_matrix(y_test, test_predictions, labels=[0, 1, 2])}"
            )

    path_keras = os.path.join(MODEL_DIR, "mobilenet_tembakau.keras")
    model.save(path_keras)
    print(f"\n-> Model format .keras BERHASIL disimpan ke: {path_keras}")

    print("=" * 65)
    akurasi_akhir = history.history['accuracy'][-1] * 100.0
    print(f"[SUKSES BESAR] Pelatihan Selesai! Akurasi Data Latih: {akurasi_akhir:.2f}%")
    print(f"Folder Penyimpanan Model : {MODEL_DIR}")
    print("Otak AI sudah siap di memori dan siap digunakan oleh web app.py!")
    print("=" * 65)

if __name__ == "__main__":
    jalankan_pelatihan()