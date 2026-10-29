# Jaringan Syaraf Tiruan (JST) - Klasifikasi Telur

Proyek ini adalah sistem klasifikasi citra berbasis deep learning untuk mengidentifikasi jenis telur ayam, yaitu:

- Telur ayam arab
- Telur ayam kampung
- Telur horn

Model dilatih menggunakan arsitektur Convolutional Neural Network (CNN), dengan pendekatan transfer learning menggunakan MobileNetV2 pada salah satu versi modelnya. Tujuan dari proyek ini adalah untuk memprediksi kelas telur dari gambar input baik melalui file gambar, GUI, maupun kamera real-time.

## Tujuan Proyek

Proyek ini dibuat untuk:

- membangun model klasifikasi citra telur,
- mempersiapkan dataset untuk training/validasi/testing,
- melatih model CNN agar mampu membedakan jenis telur berdasarkan tekstur dan bentuk citra,
- menyediakan antarmuka prediksi yang mudah digunakan.

## Teknologi yang Digunakan

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Tkinter
- Pillow (PIL)

## Struktur Proyek

```text
JST_Telur/
├── train/                  # dataset training
│   ├── arab/
│   ├── horn/
│   └── kampung/
├── valid/                  # dataset validasi
│   ├── arab/
│   ├── horn/
│   └── kampung/
├── test/                   # dataset testing
│   ├── arab/
│   ├── horn/
│   └── kampung/
├── data.yaml               # metadata dataset Roboflow
├── requirements.txt        # daftar dependency Python
├── prepare_dataset.py      # preprocessing dataset gambar
├── model.py                # arsitektur CNN custom
├── train.py                # training CNN custom
├── train_mobilenet.py      # training dengan MobileNetV2
├── predict_gui.py          # GUI untuk prediksi gambar manual
├── realtime_camera.py      # prediksi real-time via webcam
├── test_eval.py            # evaluasi model pada dataset test
├── confusion_matrix.py     # plot confusion matrix
├── cnn_telur.h5            # model hasil training custom CNN
├── cnn_telur_mobilenet.h5  # model hasil transfer learning MobileNetV2
├── Figure_1.png            # contoh visualisasi hasil
├── Figure_2.png
├── Figure_3.png
├── trainMA.png             # visualisasi training
├── trainML.png
├── Manual Prosedur.docx    # dokumentasi prosedur manual
└── README.md               # dokumentasi proyek
```

## Dataset

Dataset yang digunakan terdiri dari 3 kelas telur:

- `arab`
- `horn`
- `kampung`

Setiap folder dataset berisi gambar RGB dengan ukuran input 224x224. Proses preprocessing pada dataset dilakukan di `prepare_dataset.py`, di mana gambar dibaca, diresize, dan dinormalisasi ke rentang 0–1 sebelum masuk ke model.

## Model yang Dibangun

Ada dua pendekatan model yang tersedia:

1. Model custom CNN di `model.py`
   - Input: (224, 224, 3)
   - Layer Conv2D → MaxPooling → Conv2D → MaxPooling → Conv2D → MaxPooling
   - Flatten → Dense → Dropout → Dense output 3 kelas

2. Transfer learning MobileNetV2 di `train_mobilenet.py`
   - Menggunakan `MobileNetV2` dengan `weights="imagenet"`
   - `include_top=False`
   - Menambahkan GlobalAveragePooling2D dan Dense output 3 kelas

## Cara Menjalankan Proyek

### 1. Membuat environment virtual

```bash
python -m venv .venv
```

Pada Windows:

```bash
.venv\Scripts\activate
```

### 2. Install dependency

```bash
pip install -r requirements.txt
```

### 3. Melatih model custom CNN

```bash
python train.py
```

Model akan disimpan dalam file:

```text
cnn_telur.h5
```

### 4. Melatih model MobileNetV2

```bash
python train_mobilenet.py
```

Model akan disimpan dalam file:

```text
cnn_telur_mobilenet.h5
```

## Prediksi Gambar Manual

Untuk mengecek hasil prediksi melalui file gambar:

```bash
python predict_gui.py
```

Program ini menampilkan GUI Tkinter dengan tombol untuk memilih gambar. Setelah gambar dipilih, model akan memprediksi kelas telur dan menampilkan confidence score.

## Prediksi Real-Time dengan Kamera

Untuk prediksi langsung menggunakan webcam:

```bash
python realtime_camera.py
```

Program akan membuka kamera dan menampilkan label prediksi pada frame video secara real-time. Tekan `q` untuk keluar.

## Evaluasi Model

Untuk mengecek akurasi model pada dataset uji:

```bash
python test_eval.py
```

Script ini akan memuat model dan menampilkan nilai `loss` dan `accuracy` dari dataset test.

## Confusion Matrix

Untuk melihat visualisasi confusion matrix:

```bash
python confusion_matrix.py
```

Hasilnya akan berupa grafik matriks kebingungan yang menunjukkan seberapa baik model membedakan masing-masing kelas.

## Catatan Penting

- Ukuran input model adalah 224x224.
- Kelas harus konsisten dengan urutan pada variabel `CLASSES` di program.
- Nama folder dataset (`arab`, `horn`, `kampung`) harus sesuai dengan konfigurasi program agar label dapat terbaca dengan benar.
- Mesin yang digunakan harus memiliki dependency dari TensorFlow yang kompatibel dengan versi Python dan sistem operasi.

## Ringkasan

Proyek ini adalah contoh implementasi klasifikasi citra telur menggunakan jaringan saraf konvolusional. Model dibuat untuk mengotomatisasi proses identifikasi jenis telur dengan pendekatan berbasis image recognition, sehingga dapat diterapkan dalam sistem komputer vision yang lebih luas di masa depan.
