# Prediksi Kelulusan Mahasiswa Tepat Waktu

**UAS Kecerdasan Buatan** – Teknik Elektro  
Dosen: Dr. Raden Arief Setyawan, ST., MT.  
Tanggal: 21 Juni 2026

## Deskripsi Proyek
Proyek ini menggunakan **Machine Learning** (Random Forest) untuk memprediksi apakah seorang mahasiswa akan lulus tepat waktu (≤4 tahun) atau tidak berdasarkan fitur akademik dan non-akademik.

## Dataset
- **Jumlah data:** 500 baris (memenuhi syarat minimal 300)
- **Fitur:**
  - IP Semester 1, 2, 3, 4
  - Jumlah mata kuliah tidak lulus
  - Keaktifan organisasi (0/1)
  - Status beasiswa (0/1)
  - Persentase kehadiran
- **Target:** Lulus tepat waktu (1) / Tidak (0)
- **Distribusi kelas:** 377 (tidak tepat waktu), 123 (tepat waktu)

## Metode
- **Preprocessing:** Standard scaling (StandardScaler)
- **Model:** Random Forest dengan `class_weight='balanced'` (mengatasi ketidakseimbangan kelas)
- **Evaluasi:** Akurasi, classification report, confusion matrix

## Hasil
- **Akurasi:** 100%
- **Classification report:**
  - Precision, recall, f1-score = 1.00 untuk kedua kelas
- **Confusion matrix:**  
  ![Confusion Matrix](confusion_matrix.png)

## Cara Menjalankan
1. Buka file `prediksi_kelulusan.ipynb` di Google Colab atau Jupyter Notebook.
2. Jalankan semua sel secara berurutan.
3. Pastikan library terinstal: `pandas`, `numpy`, `scikit-learn`, `matplotlib`.

## Video Demo
[Klik di sini untuk menonton demo](https://youtu.be/___GANTI_DENGAN_LINK_ANDA___)

## Slide Presentasi
[Link slide (Google Drive)](https://drive.google.com/___GANTI_DENGAN_LINK_ANDA___)

## Video Presentasi

## Laporan

## Kontributor
Nama: Muhammad Naufal Sa'd  
NIM: 235060301111026 

---
