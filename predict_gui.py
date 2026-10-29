import tkinter as tk
# Tkinter digunakan untuk membuat GUI (jendela aplikasi)

from tkinter import filedialog
# Digunakan untuk membuka dialog pemilihan file (Open File)

from PIL import Image, ImageTk
# PIL (Pillow) digunakan untuk memproses dan menampilkan gambar di Tkinter

import cv2
# OpenCV digunakan untuk membaca dan memproses gambar

import numpy as np
# NumPy digunakan untuk operasi array dan manipulasi data numerik

from tensorflow.keras.models import load_model
# Digunakan untuk memuat model CNN yang sudah dilatih

# ================== KONFIGURASI ==================
IMG_SIZE = 224
# Ukuran input gambar yang sesuai dengan model CNN

CLASSES = ["arab", "horn", "kampung"]
# Daftar kelas telur
# Urutan harus SAMA dengan urutan saat training model

model = load_model("cnn_telur_mobilenet.h5")
# Memuat model CNN berbasis MobileNet yang sudah dilatih

# ================== FUNGSI PILIH GAMBAR ==================
def pilih_gambar():
    # Membuka dialog untuk memilih file gambar
    path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )

    # Jika user batal memilih gambar
    if not path:
        return

    # Membaca gambar menggunakan OpenCV
    img = cv2.imread(path)
    if img is None:
        # Jika gambar gagal dibaca
        hasil_label.config(text="Gambar tidak bisa dibaca")
        return

    # Resize gambar ke ukuran input CNN
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Normalisasi pixel (0–255 → 0–1)
    img_norm = img_resized / 255.0

    # Menambahkan dimensi batch (1, 224, 224, 3)
    img_input = np.expand_dims(img_norm, axis=0)

    # Prediksi menggunakan model CNN
    pred = model.predict(img_input)

    # Mengambil index kelas dengan probabilitas tertinggi
    idx = np.argmax(pred)

    # Menampilkan hasil prediksi dan confidence
    hasil_label.config(
        text=f"Prediksi: {CLASSES[idx]}\nConfidence: {pred[0][idx]:.2f}"
    )

    # ================== TAMPILKAN GAMBAR DI GUI ==================

    # Konversi warna BGR (OpenCV) ke RGB (PIL)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Mengubah array ke format PIL Image
    img_pil = Image.fromarray(img_rgb)

    # Resize gambar agar pas di GUI
    img_pil = img_pil.resize((300, 300))

    # Konversi ke format yang bisa ditampilkan Tkinter
    img_tk = ImageTk.PhotoImage(img_pil)

    # Menampilkan gambar di label
    img_label.config(image=img_tk)

    # Menyimpan referensi agar gambar tidak hilang
    img_label.image = img_tk


# ================== MEMBUAT GUI ==================
root = tk.Tk()
# Membuat window utama GUI

root.title("Klasifikasi Telur Ayam (CNN)")
# Judul window

root.geometry("400x500")
# Ukuran window GUI

# Tombol untuk memilih gambar
btn = tk.Button(root, text="Pilih Gambar Telur", command=pilih_gambar)
btn.pack(pady=10)

# Label untuk menampilkan gambar
img_label = tk.Label(root)
img_label.pack()

# Label untuk menampilkan hasil prediksi
hasil_label = tk.Label(root, text="Prediksi: -", font=("Arial", 12))
hasil_label.pack(pady=10)

# Menjalankan GUI
root.mainloop()
