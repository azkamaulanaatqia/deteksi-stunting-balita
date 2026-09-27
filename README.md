# Deteksi Dini Risiko Stunting Balita

Proyek ini merupakan tugas akhir mata kuliah Artificial Intelligence (DTETI, Universitas Gadjah Mada) yang mengimplementasikan algoritma Naive Bayes untuk mengklasifikasikan risiko stunting pada balita berdasarkan data antropometri posyandu.

## Tentang Proyek

Stunting masih menjadi salah satu masalah gizi utama di Indonesia, dan deteksi dini sering terkendala oleh keterbatasan tenaga kesehatan untuk melakukan penilaian manual di lapangan. Proyek ini mencoba membangun alat skrining sederhana yang dapat memberikan estimasi awal risiko stunting hanya dari tiga variabel yang mudah diukur: jenis kelamin, usia, dan tinggi badan anak. Model yang digunakan adalah Gaussian Naive Bayes, dipilih karena ringan secara komputasi, tidak membutuhkan data pelatihan dalam jumlah besar, dan menghasilkan keluaran berupa probabilitas yang cocok untuk konteks skrining (bukan diagnosis definitif).

## Dataset

Data yang digunakan berasal dari *Stunting and Nutritional Status of Toddler from Jeneponto Regency, South Sulawesi, Indonesia* (Mendeley Data, DOI: [10.17632/wzwpc9j5bx](https://doi.org/10.17632/wzwpc9j5bx)), berisi 40.071 hasil pengukuran balita yang dikumpulkan dari posyandu di Kabupaten Jeneponto sepanjang tahun 2021–2024.

Data mentah ini ternyata cukup bermasalah saat pertama kali diperiksa: sekitar separuh nilai berat badan tersimpan dalam format tanggal akibat kesalahan konversi otomatis di spreadsheet, dan beberapa baris memiliki satuan berat yang tertukar (gram alih-alih kilogram). Kedua masalah ini ditelusuri dan diperbaiki sebelum data digunakan lebih lanjut, bukan sekadar dibuang, karena baris yang bermasalah ternyata tidak tersebar acak terhadap label targetnya — membuangnya begitu saja akan mendistorsi proporsi kelas secara signifikan. Setelah proses pembersihan dan penghapusan duplikat, dataset final berisi 30.402 baris.

## Pendekatan Model

Pengujian awal dengan fitur mentah (usia dan tinggi badan sebagai variabel terpisah) ternyata memberi hasil yang mengecewakan, bahkan sedikit di bawah baseline sederhana yang hanya menebak kelas mayoritas. Setelah ditelusuri, penyebabnya cukup mendasar: status stunting sebenarnya ditentukan oleh tinggi badan *relatif terhadap* usia, bukan tinggi badan sebagai angka mutlak, sehingga memperlakukan keduanya sebagai fitur independen (sesuai asumsi Naive Bayes) menghilangkan informasi yang justru paling menentukan.

Sebagai perbaikan, tinggi badan distandardisasi terhadap rata-rata dan simpangan baku pada kelompok usianya masing-masing (dibagi ke dalam lima rentang: 0–11, 12–23, 24–35, 36–47, dan 48–59 bulan), menghasilkan fitur `Height_std` yang secara konsep mendekati Z-score tinggi-badan-menurut-umur. Statistik kelompok usia ini dihitung khusus dari data latih untuk menghindari kebocoran informasi ke data uji. Model final menggunakan dua fitur, `Gender_Code` dan `Height_std`, dan mencapai akurasi 0,81 pada data uji — jauh meningkat dari 0,62 pada versi awal dengan fitur mentah.

## Struktur Folder
├── data/
│ └── stunting_jeneponto_preprocessed.csv
├── notebook/
│ └── train_model.py
├── model/
│ ├── stunting_nb_model.pkl
│ └── encoding_map.json
├── app/
│ └── app.py
├── requirements.txt
└── README.md


## Cara Menjalankan

Instal dependensi terlebih dahulu:

```bash
pip install -r requirements.txt
```

Model yang sudah dilatih tersedia langsung di folder `model/`, jadi tahap training bersifat opsional dan hanya perlu dijalankan ulang jika ingin memverifikasi atau memodifikasi proses pelatihannya:

```bash
cd notebook
python train_model.py
```

Untuk menjalankan aplikasi demo, jalankan perintah berikut dari folder root project (bukan dari dalam folder `app/`):

```bash
streamlit run app/app.py
```

Aplikasi akan terbuka di browser pada local host

## Hasil Evaluasi

| Metrik | Nilai |
|---|---|
| Accuracy | 0,8066 |
| Precision (Stunted) | 0,76 |
| Recall (Stunted) | 0,70 |
| F1-score (Stunted) | 0,73 |
| Baseline (prediksi kelas mayoritas) | 0,6323 |

## Limitasi

Model ini masih memiliki beberapa keterbatasan yang perlu diperhatikan. Asumsi independensi kondisional pada Naive Bayes tetap dilanggar oleh korelasi alami antar variabel antropometri, meski dampaknya sudah dikurangi lewat rekayasa fitur `Height_std`. Standardisasi usia yang digunakan juga dihitung dari statistik dataset ini sendiri, bukan dari tabel rujukan pertumbuhan resmi WHO, sehingga hasilnya bisa jadi tidak sepenuhnya konsisten dengan standar klinis yang berlaku umum. Variabel berat badan pada akhirnya tidak disertakan pada model final karena tidak memberikan peningkatan performa yang konsisten selama pengujian. Yang terpenting, alat ini dirancang sebagai skrining awal untuk membantu identifikasi risiko, bukan sebagai pengganti diagnosis oleh tenaga kesehatan profesional.

## Sumber Data

Dataset: *Stunting and Nutritional Status of Toddler from Jeneponto Regency, South Sulawesi, Indonesia*, Mendeley Data, Universitas Negeri Makassar. [https://data.mendeley.com/datasets/wzwpc9j5bx](https://data.mendeley.com/datasets/wzwpc9j5bx)