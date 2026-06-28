# Student Burnout Intelligence

Proyek data science end-to-end untuk menganalisis dan memprediksi tingkat risiko burnout mahasiswa berdasarkan pola penggunaan Generative AI (GenAI), kebiasaan belajar, profil akademik, ketergantungan terhadap AI, dan kecemasan saat ujian.

Proyek dikembangkan untuk **GWE 2026 Data Science Challenge** dengan subtema **Education** dan **Risk Prediction** menggunakan metode **CRISP-DM**.

## Tautan Proyek

- **Aplikasi Streamlit:** [Tambahkan tautan deployment](TAUTAN_STREAMLIT)
- **Repository GitHub:** [Tambahkan tautan repository](TAUTAN_GITHUB)
- **Sumber dataset:** [Tambahkan tautan sumber dataset](TAUTAN_DATASET)

> Ganti seluruh placeholder tautan sebelum repository dikumpulkan.

## Identitas Tim

- **Nama tim:** WhatIf
- **Anggota:** Ihwan Fajar Maulana
- **Program studi:** S1 Sistem Informasi

## Latar Belakang

GenAI semakin banyak digunakan mahasiswa untuk mencari ide, merangkum materi, menyusun tulisan, melakukan debugging, dan memperoleh jawaban. Pemanfaatan ini dapat membantu proses akademik, tetapi pola penggunaan yang tidak seimbang juga dapat berkaitan dengan ketergantungan AI, kecemasan saat ujian, berkurangnya proses belajar mandiri, dan risiko burnout.

Proyek ini dibangun untuk mengeksplorasi pola tersebut dan menyediakan prediksi kategori risiko burnout secara interaktif. Hasil aplikasi merupakan dukungan analisis berbasis data dan **bukan diagnosis medis atau psikologis**.

## Tujuan

1. Menganalisis karakteristik mahasiswa berdasarkan penggunaan GenAI dan kebiasaan belajar.
2. Mengidentifikasi pola yang berkaitan dengan risiko burnout.
3. Membandingkan beberapa algoritma klasifikasi multikelas.
4. Memprediksi kategori risiko `Low`, `Medium`, atau `High`.
5. Menyediakan EDA dan prediksi real-time melalui Streamlit.

## Dataset

Dataset terdiri dari **50.000 baris dan 16 variabel**. Target yang digunakan adalah `Burnout_Risk_Level`.

Distribusi target:

| Kategori | Jumlah | Persentase |
|---|---:|---:|
| Low | 16.369 | 32,74% |
| Medium | 21.144 | 42,29% |
| High | 12.487 | 24,97% |

Beberapa variabel utama:

- `Weekly_GenAI_Hours`
- `Traditional_Study_Hours`
- `Perceived_AI_Dependency`
- `Anxiety_Level_During_Exams`
- `Pre_Semester_GPA`
- `Major_Category`
- `Year_of_Study`
- `Primary_Use_Case`

`Student_ID` tidak digunakan sebagai fitur karena hanya berfungsi sebagai identifier. `Post_Semester_GPA` dan `Skill_Retention_Score` juga tidak digunakan untuk menghindari data leakage dalam skenario sistem peringatan dini.

## Metodologi CRISP-DM

### 1. Business Understanding

Menentukan masalah, tujuan, manfaat, batasan, dan ruang lingkup prediksi risiko burnout mahasiswa.

### 2. Data Understanding

Mengidentifikasi struktur dataset, tipe variabel, statistik deskriptif, distribusi target, serta pola awal data.

### 3. Data Preparation

- Pemeriksaan missing value dan data duplikat.
- Pemeriksaan outlier menggunakan metode IQR.
- Mempertahankan outlier yang masih berada dalam rentang logis.
- Feature selection untuk menghapus identifier dan fitur berpotensi leakage.
- Feature engineering.
- One-Hot Encoding untuk fitur kategorikal.
- StandardScaler untuk fitur numerik.
- Train-test split dengan rasio 80:20 dan stratifikasi target.

Fitur hasil rekayasa:

- `GenAI_Study_Ratio`: rasio penggunaan GenAI terhadap belajar tradisional.
- `GenAI_Usage_Category`: kategori intensitas penggunaan GenAI.

### 4. Modeling

Model yang dibandingkan:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Random Forest hasil hyperparameter tuning

Preprocessing dan algoritma digabungkan dalam satu `Pipeline` agar transformasi data konsisten pada training, testing, dan aplikasi Streamlit.

### 5. Evaluation

Metrik evaluasi yang digunakan:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Recall kelas High
- Classification report
- Confusion matrix

### 6. Deployment

Model dan preprocessing disimpan dalam file pipeline `burnout_model.pkl`, kemudian digunakan oleh aplikasi Streamlit untuk prediksi real-time.

## Hasil Eksplorasi Data

Rata-rata berdasarkan kategori risiko:

| Variabel | Low | Medium | High |
|---|---:|---:|---:|
| Penggunaan GenAI per minggu | 4,64 | 7,35 | 15,21 |
| Ketergantungan AI | 2,82 | 3,36 | 4,64 |
| Kecemasan saat ujian | 3,93 | 4,17 | 4,89 |
| Belajar tradisional per minggu | 11,97 | 11,29 | 10,08 |

Insight utama:

- Kelompok risiko High memiliki rata-rata penggunaan GenAI paling tinggi.
- Kelompok risiko yang lebih tinggi juga memiliki rata-rata ketergantungan AI dan kecemasan yang lebih tinggi.
- Jam belajar tradisional cenderung lebih rendah pada kelompok High.
- Korelasi tertinggi ditemukan antara penggunaan GenAI dan ketergantungan AI, yaitu sekitar `0,67`.
- Insight menunjukkan hubungan pada dataset dan tidak membuktikan hubungan sebab-akibat.

## Hasil Model

| Model | Accuracy | Precision Macro | Recall Macro | F1 Macro | Recall High |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0,5381 | 0,5684 | 0,5251 | 0,5381 | 0,4666 |
| Gradient Boosting | 0,5348 | 0,5642 | 0,5230 | 0,5357 | 0,4734 |
| Random Forest Tuning | 0,5316 | 0,5617 | 0,5187 | 0,5315 | 0,4666 |
| Random Forest | 0,5152 | 0,5397 | 0,5076 | 0,5185 | 0,4786 |

**Logistic Regression** digunakan sebagai model final karena memperoleh Macro F1-score tertinggi pada eksperimen saat ini. Accuracy model sebesar **53,81%**, lebih tinggi daripada majority baseline sebesar **42,29%**.

Performa model masih tergolong moderat. Model digunakan sebagai prototipe analisis awal dan belum layak menjadi dasar tunggal untuk keputusan terkait kesehatan mental mahasiswa.

## Fitur Aplikasi

### Beranda

- Deskripsi proyek
- Latar belakang masalah
- Tujuan proyek
- Ringkasan dataset dan model

### Dashboard EDA

- Filter berdasarkan jurusan, tahun studi, kategori risiko, dan GPA
- KPI data terpilih
- Distribusi risiko burnout
- Analisis penggunaan GenAI dan kebiasaan belajar
- Heatmap korelasi
- Tabel data hasil filter

### Prediksi Burnout

- Input profil akademik dan kebiasaan belajar
- Prediksi kategori `Low`, `Medium`, atau `High`
- Probabilitas prediksi setiap kelas
- Rekomendasi awal berdasarkan hasil model

### Dokumentasi

- Tahapan CRISP-DM
- Penjelasan fitur dan model
- Tabel metrik evaluasi
- Cara penggunaan aplikasi
- Informasi tim

## Struktur Repository

```text
student-burnout-intelligence/
|-- README.md
|-- requirements.txt
|-- app.py
|-- Analisis_AI_Student_Impact.ipynb
|-- ai_student_impact_dataset (1).csv
|-- burnout_model.pkl
|-- label_encoder.pkl
|-- model_evaluation.csv
`-- .gitignore
```

## Instalasi

### 1. Clone Repository

```bash
git clone TAUTAN_GITHUB
cd student-burnout-intelligence
```

### 2. Buat Virtual Environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Linux atau macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Instal Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi lokal akan tersedia melalui alamat yang ditampilkan pada terminal, biasanya `http://localhost:8501`.

## File Model

- `burnout_model.pkl`: pipeline preprocessing dan model final.
- `label_encoder.pkl`: encoder untuk mengubah hasil numerik menjadi label risiko.
- `model_evaluation.csv`: hasil perbandingan metrik model.

File model harus dibuat ulang apabila struktur fitur, preprocessing, atau versi `scikit-learn` berubah.

## Keterbatasan

- Performa model masih moderat.
- Recall kategori High masih terbatas.
- Distribusi fitur antarkategori masih banyak bertumpang tindih.
- Hasil sangat bergantung pada kualitas dan representasi dataset.
- Model belum divalidasi pada populasi mahasiswa dari institusi lain.
- Prediksi tidak dapat dianggap sebagai diagnosis kesehatan mental.

## Pengembangan Berikutnya

- Melakukan pemilihan model menggunakan cross-validation pada data training.
- Melakukan tuning seluruh model menggunakan Macro F1-score.
- Menambahkan variabel seperti kualitas tidur, beban tugas, dan dukungan sosial.
- Melakukan validasi eksternal.
- Menambahkan interpretabilitas model seperti SHAP atau LIME.
- Melakukan pemantauan performa model setelah deployment.

## Penggunaan AI

AI generatif digunakan sebagai alat bantu dalam:

- Menyusun dan memperbaiki dokumentasi proyek.
- Menjelaskan konsep data science dan machine learning.
- Membantu proses debugging dan pengembangan antarmuka Streamlit.
- Memberikan saran visualisasi, evaluasi model, dan struktur presentasi.

Seluruh kode, analisis, output, dan dokumentasi telah ditinjau kembali oleh pengembang. Keputusan akhir terkait preprocessing, pemodelan, evaluasi, dan interpretasi tetap menjadi tanggung jawab pengembang.

## Lisensi dan Penggunaan

Proyek ini dikembangkan untuk tujuan pembelajaran dan kompetisi GWE 2026. Penggunaan dataset mengikuti lisensi dari sumber aslinya.
