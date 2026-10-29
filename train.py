from prepare_dataset import load_data
# Mengimpor fungsi load_data untuk membaca dan menyiapkan dataset (gambar + label)

from model import build_model
# Mengimpor fungsi build_model untuk membuat arsitektur CNN

import matplotlib.pyplot as plt
# Matplotlib digunakan untuk membuat grafik akurasi dan loss

# ================== LOAD DATASET ==================
X_train, y_train = load_data("train")
# Memuat data training dari folder "train"
# X_train = gambar
# y_train = label kelas

X_val, y_val = load_data("valid")
# Memuat data validasi dari folder "valid"
# Digunakan untuk mengecek performa model selama training

# ================== BUILD MODEL ==================
model = build_model()
# Membuat model CNN menggunakan arsitektur yang didefinisikan di model.py

# ================== TRAINING MODEL ==================
history = model.fit(
    X_train, y_train,
    # Data training (gambar dan label)

    validation_data=(X_val, y_val),
    # Data validasi untuk mengecek overfitting

    epochs=10,
    # Jumlah pengulangan training
    # 1 epoch = seluruh data training dipelajari sekali

    batch_size=16
    # Jumlah gambar yang diproses sekali update bobot
)

# ================== SAVE MODEL ==================
model.save("cnn_telur.h5")
# Menyimpan model CNN hasil training ke file
# File ini nanti dipakai untuk predict, GUI, dan kamera

# ================== PLOT AKURASI ==================
plt.figure()
# Membuat figure baru

plt.plot(history.history["accuracy"])
# Menampilkan akurasi data training tiap epoch

plt.plot(history.history["val_accuracy"])
# Menampilkan akurasi data validasi tiap epoch

plt.title("Model Accuracy")
# Judul grafik

plt.xlabel("Epoch")
# Label sumbu X

plt.ylabel("Accuracy")
# Label sumbu Y

plt.legend(["Train", "Validation"])
# Keterangan garis grafik

plt.show()
# Menampilkan grafik akurasi

# ================== PLOT LOSS ==================
plt.figure()
# Membuat figure baru

plt.plot(history.history["loss"])
# Menampilkan nilai loss data training

plt.plot(history.history["val_loss"])
# Menampilkan nilai loss data validasi

plt.title("Model Loss")
# Judul grafik

plt.xlabel("Epoch")
# Label sumbu X

plt.ylabel("Loss")
# Label sumbu Y

plt.legend(["Train", "Validation"])
# Keterangan grafik

plt.show()
# Menampilkan grafik loss
