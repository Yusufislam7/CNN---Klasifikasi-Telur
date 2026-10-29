import cv2
# OpenCV digunakan untuk mengakses kamera dan menampilkan video real-time

import numpy as np
# NumPy digunakan untuk manipulasi array dan data numerik

from tensorflow.keras.models import load_model
# Digunakan untuk memuat model CNN yang sudah dilatih

# ================== KONFIGURASI ==================
IMG_SIZE = 224
# Ukuran input gambar yang sesuai dengan model CNN (MobileNet)

CLASSES = ["arab", "horn", "kampung"]
# Daftar kelas telur
# Urutan HARUS sama dengan urutan training

model = load_model("cnn_telur_mobilenet.h5")
# Memuat model CNN berbasis MobileNet yang telah dilatih

# ================== AKSES KAMERA ==================
cap = cv2.VideoCapture(0)
# Membuka webcam
# 0 = kamera utama laptop

if not cap.isOpened():
    # Mengecek apakah kamera berhasil dibuka
    print("Kamera tidak bisa dibuka")
    exit()

# ================== LOOP REAL-TIME ==================
while True:
    ret, frame = cap.read()
    # Membaca satu frame dari kamera
    # ret = status berhasil/gagal
    # frame = gambar dari kamera

    if not ret:
        # Jika frame gagal dibaca
        break

    # ================== PREPROCESSING ==================
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    # Resize frame agar sesuai input CNN

    img = img / 255.0
    # Normalisasi pixel (0–255 → 0–1)

    img = np.expand_dims(img, axis=0)
    # Menambahkan dimensi batch
    # Shape jadi: (1, 224, 224, 3)

    # ================== PREDIKSI ==================
    pred = model.predict(img, verbose=0)
    # Model melakukan prediksi kelas

    idx = np.argmax(pred)
    # Mengambil index kelas dengan probabilitas tertinggi

    label = CLASSES[idx]
    # Mengubah index menjadi nama kelas

    confidence = pred[0][idx]
    # Mengambil nilai confidence dari prediksi

    text = f"{label} ({confidence:.2f})"
    # Membuat teks hasil prediksi

    # ================== TAMPILKAN TEKS ==================
    cv2.putText(
        frame,
        text,                     # Teks prediksi
        (20, 40),                 # Posisi teks
        cv2.FONT_HERSHEY_SIMPLEX, # Jenis font
        1,                        # Ukuran font
        (0, 255, 0),              # Warna teks (Hijau)
        2                         # Ketebalan teks
    )

    # ================== TAMPILKAN VIDEO ==================
    cv2.imshow("Real-Time Klasifikasi Telur", frame)
    # Menampilkan video real-time dengan hasil prediksi

    # ================== KELUAR PROGRAM ==================
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Tekan tombol 'q' untuk keluar
        break

# ================== BERSIHKAN RESOURCE ==================
cap.release()
# Melepaskan kamera

cv2.destroyAllWindows()
# Menutup semua jendela OpenCV
