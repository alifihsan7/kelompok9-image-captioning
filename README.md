# Kelompok 9 — Image Captioning

Proyek mata kuliah untuk membuat sistem yang menghasilkan deskripsi singkat bahasa Inggris dari sebuah gambar menggunakan CNN feature extractor dan LSTM decoder.

Contoh target keluaran: `a dog is running on the grass`.

## Status proyek

Proyek berada pada tahap persiapan. Arsitektur di bawah merupakan rancangan awal berdasarkan workflow kelompok. Kode, petunjuk menjalankan, dependency, dan hasil evaluasi akan diperbarui setelah implementasi diperiksa.

**Penggunaan encoder pretrained masih perlu dikonfirmasi kepada dosen** agar sesuai dengan ketentuan *from scratch*.

## Rancangan model

- **Dataset:** Flickr8k, dengan caption bahasa Inggris.
- **Encoder CNN:** ResNet50 pretrained ImageNet, dibekukan dan tanpa classifier terakhir.
- **Fitur gambar:** 2048 dimensi.
- **Decoder:** embedding 256 dimensi dan LSTM satu lapis dengan hidden size 512.
- **Penghubung:** dua proyeksi terpisah untuk hidden state dan cell state awal LSTM.
- **Training:** decoder dilatih dari bobot acak dengan teacher forcing.
- **Prediksi awal:** greedy decoding; beam search sebagai pengembangan.

```text
Gambar → ResNet50 → fitur gambar → LSTM decoder → caption
```

Encoder menggunakan bobot pretrained. Karena itu, pendekatan ini tidak dinyatakan sebagai seluruh model dilatih dari nol.

## Pembagian tugas

| Anggota | Tanggung jawab utama |
|---|---|
| 1 — Prompt Engineer dan Ketua | Kontrak arsitektur, decoder, koordinasi integrasi, dan decoding |
| 2 — Data Pipeline | Dataset, preprocessing, vocabulary, split, dan ekstraksi fitur |
| 3 — Training dan Tuning | Training loop, checkpoint, eksperimen, dan evaluasi |
| 4 — Dokumentasi dan Error Handling | Kompilasi log LLM, koordinasi debugging, demo, dan laporan |

## Tempat kerja

| Tempat | Fungsi |
|---|---|
| Notebooks di Gemini | Sumber proyek dan interaksi LLM menggunakan akun kelompok |
| Google Drive | File kerja, dataset, fitur, checkpoint, hasil, dan log |
| Google Colab | Menjalankan notebook serta kode Python |
| GitHub | Menyimpan versi kode/notebook yang diperiksa dan riwayat kontribusi |

Setiap anggota menyebut nama, peran, dan target saat memulai sesi LLM. Keputusan antarmodul dicatat bersama agar implementasi konsisten.

## Struktur repository yang direncanakan

```text
kelompok9-image-captioning/
├── README.md
├── requirements.txt
├── .gitignore
├── src/          # dataset.py, model.py, train_utils.py
├── notebooks/    # 01_data, 02_model, 03_train, 04_demo
├── docs/         # kontrak arsitektur dan dokumentasi teknis
└── logs/         # dokumentasi interaksi LLM
```

Struktur ini merupakan target penataan; file dibuat ketika mulai dikerjakan. Dataset, fitur, dan checkpoint besar disimpan di Drive dalam folder `Kelompok9/`, dengan subfolder `data/`, `features/`, `src/`, `notebooks/`, `checkpoints/`, `results/`, dan `logs/`.

## Alur pengerjaan

1. Siapkan akses GitHub, folder Drive, dan notebook Gemini bersama.
2. Sepakati arsitektur, kontrak data, serta status izin pretrained.
3. Kerjakan pipeline data, decoder, training loop, dan dokumentasi sesuai pembagian tugas.
4. Periksa modul dengan data dummy, kemudian integrasikan menggunakan satu batch nyata.
5. Latih baseline, lakukan eksperimen terkontrol, dan pilih konfigurasi berdasarkan validation.
6. Evaluasi pilihan final pada test, siapkan demo, dan selesaikan laporan.

File kerja disimpan di Drive dan dijalankan melalui Colab. Versi kode serta notebook yang selesai diperiksa disimpan ke GitHub. File `.py` perlu diunggah secara eksplisit; menyimpan notebook saja tidak otomatis menyimpan file Python yang dibuatnya.

## Evaluasi dan dokumentasi

Evaluasi direncanakan menggunakan BLEU-1 sampai BLEU-4, dilengkapi pemeriksaan kesesuaian caption terhadap gambar. Hasil belum tersedia dan akan dicatat setelah eksperimen dijalankan.

Setiap sesi LLM didokumentasikan melalui prompt, alasan keputusan, ringkasan respons, pemeriksaan aktual, serta error dan perbaikannya jika ada. Riwayat percakapan disimpan sebagai tautan atau ekspor. Kontribusi kode dan dokumentasi dicatat melalui commit masing-masing anggota.

## Acuan

- Panduan Proyek 1 — Vibe Coding dari dosen.
- Workflow Kelompok 9 — Image Captioning (CNN + LSTM).
- Panduan tahapan kerja kelompok yang disusun berdasarkan workflow.
