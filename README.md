# DINO-DETR Player and Ball Detection Training (FP32 Baseline)

Proyek ini bertujuan melatih arsitektur **DINO-DETR** dalam presisi **FP32 murni** (baseline akurasi) untuk mendeteksi dua objek utama: pemain sepak bola (`person`) dan bola (`ball`).

---

## 🛠️ Persiapan Lingkungan & Instalasi

### 1. Memasang Dependensi
Pasang seluruh paket python yang terdaftar di dalam `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Kloning Repositori DINO Resmi
Kloning repositori DINO resmi dari IDEA-Research ke dalam folder bernama `official_dino`:
```bash
git clone https://github.com/IDEA-Research/DINO.git official_dino
```

### 3. Kompilasi CUDA Operators (Ops)
Model DINO membutuhkan operator C++/CUDA kustom untuk melakukan *Multi-Scale Deformable Attention*. 

Jalankan skrip build di dalam folder ops:
```bash
cd official_dino/models/dino/ops
python setup.py build develop
cd ../../../../
```

> [!NOTE]
> **Autofallback**: Jika proses kompilasi di atas gagal (misalnya karena tidak adanya compiler MSVC C++ `cl.exe` di Windows), model DINO ini otomatis akan mendeteksi kegagalan tersebut dan menggunakan **Pure PyTorch Fallback (`F.grid_sample`)** yang telah diprogram secara native di dalam `model.py`. Anda tidak perlu khawatir jika build CUDA gagal, sistem akan tetap berjalan dengan fallback tersebut.

---

## 📂 Langkah Pelatihan

### 1. Konversi Format YOLO ke COCO JSON
Sebelum melatih model, konversi dataset YOLO Anda yang terletak di dalam `./merged_yolo_person_ball` menjadi format COCO:
```bash
python convert_yolo_to_coco.py
```
* Skrip ini akan secara otomatis memisahkan data `train` dan `val` sesuai split direktori dataset Anda.
* Hasil konversi akan tersimpan sebagai `annotations_train.json` dan `annotations_val.json`.
* **Sanity Check**: Periksa 5 gambar hasil plotting bounding box di dalam direktori `./sanity_checks` untuk memastikan letak anotasi sudah presisi di atas pemain dan bola.

### 2. Uji Sanity-Check Overfitting (10 Gambar)
Guna memastikan pipeline backpropagation, kalkulasi loss, dan aliran gradien berjalan dengan benar, jalankan tes overfitting pada 10 gambar selama 50 epoch:
```bash
python train.py --overfit-check
```
* Skrip ini membatasi dataset hanya pada 10 gambar pertama.
* Amati output loss. Tes dinyatakan **[SUCCESS]** jika loss pelatihan turun drastis (minimal berkurang 50% dari loss awal).

### 3. Eksekusi Pelatihan Baseline (FP32)
Untuk memulai pelatihan penuh menggunakan semua gambar dengan parameter FP32 murni:
```bash
python train.py --epochs 12 --batch-size 4 --lr 1e-4
```

> [!TIP]
> **VRAM & OOM Fallback**: Pelatihan FP32 penuh pada arsitektur DINO-DETR membutuhkan VRAM yang cukup besar. Jika terjadi error Out-Of-Memory (OOM) saat training berjalan, script `train.py` akan menangkap error tersebut secara dinamis, mengosongkan cache VRAM, **menurunkan batch size secara otomatis setengahnya** (misal dari 4 ke 2), lalu memulai kembali latihan tanpa perlu merestart program secara manual.
