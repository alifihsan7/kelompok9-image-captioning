# Kelompok 9 — Image Captioning dengan CNN dan LSTM

Proyek ini membangun sistem yang menerima sebuah gambar dan menghasilkan deskripsi singkat dalam bahasa Inggris. Contoh target keluaran: `a dog is running on the grass`.

Dokumen ini merupakan acuan proyek untuk notebook Gemini dan repository kelompok. Konfigurasi berikut adalah baseline yang direncanakan; hasil training dan metrik diisi setelah eksperimen dijalankan.

## Pendekatan proyek

Mengikuti workflow kelompok, CNN yang digunakan adalah **ResNet50 pretrained sebagai feature extractor beku**, dipasangkan dengan **LSTM decoder yang dibangun sendiri dan dilatih dari bobot acak**.

Penggunaan encoder pretrained masih perlu dikonfirmasi kepada dosen sesuai ketentuan *from scratch*. Pendekatan ini tidak boleh dilaporkan sebagai seluruh model dilatih dari nol. Jika keputusan encoder berubah, kontrak data, kode, dan dokumen tim harus diperbarui bersama.

```text
Gambar
  → preprocessing sesuai bobot encoder
  → ResNet50 tanpa classifier terakhir
  → fitur gambar 2048 dimensi
  → dua proyeksi untuk hidden state dan cell state awal
  → LSTM decoder
  → prediksi kata secara berurutan
  → caption
```

## Pembagian tugas

| Anggota | Peran | Tanggung jawab utama |
|---|---|---|
| Anggota 1 | Prompt Engineer dan Ketua | Prompt acuan, kontrak arsitektur, `model.py`, integrasi, greedy/beam decoding, dan koordinasi submit |
| Anggota 2 | Data Pipeline | Dataset, preprocessing, vocabulary, split, ekstraksi fitur, dan `dataset.py` |
| Anggota 3 | Training dan Tuning | Training loop, checkpoint, eksperimen, grafik loss, dan evaluasi BLEU |
| Anggota 4 | Dokumentasi dan Error Handling | Kompilasi log LLM, pencatatan debugging, notebook demo, dan laporan akhir |

Setiap anggota menyimpan riwayat pekerjaannya dan memahami kode yang digunakan. Perubahan yang memengaruhi anggota lain dibahas sebelum digabungkan.

## Penggunaan Notebooks di Gemini

Gunakan satu notebook utama bernama **Kelompok 9 — Image Captioning CNN + LSTM** sebagai ruang konteks proyek.

1. Pilih **New notebook**, lalu beri nama proyek.
2. Tambahkan panduan dosen, workflow kelompok, panduan anggota 1, dan README ini melalui **Add sources**. Jika suatu format tidak diterima, gunakan ekspor PDF atau salin isi teks sesuai pilihan yang tersedia.
3. Awali setiap sesi dengan nama anggota, modul, target, dan versi kontrak.
4. Minta bantuan bertahap: rancangan, implementasi modul, pemeriksaan, debugging, lalu optimasi.
5. Simpan keputusan yang sudah disepakati ke dokumen acuan. Perbarui sumber secara eksplisit saat keputusan berubah; jangan menganggap salinan sumber otomatis mengikuti perubahan file lokal.
6. Simpan kode yang sudah ditinjau ke folder proyek dan GitHub. Simpan log sebagai tautan percakapan atau ekspor yang dapat diakses.

Notebook Gemini dipakai untuk mengorganisasi konteks dan interaksi LLM. Lingkungan training proyek yang direncanakan tetap **Google Colab**, sedangkan berkas besar dan checkpoint disimpan di **Google Drive**.

Dokumentasi fitur: [Organize your projects with notebooks in Gemini Apps](https://support.google.com/gemini/answer/16972047?hl=en).

### Prompt pembuka proyek

```text
Saya [nama], anggota [nomor] dengan peran [peran] di Kelompok 9.
Target sesi saya: [modul dan hasil yang ingin diselesaikan].

Gunakan panduan tugas dan workflow yang saya tambahkan sebagai referensi.
Proyek kami adalah Image Captioning menggunakan Python dan PyTorch.
Baseline: ResNet50 pretrained IMAGENET1K_V2 dibekukan sebagai encoder,
dan LSTM decoder dibuat sendiri serta dilatih dari bobot acak.
Dataset yang direncanakan adalah Flickr8k dengan caption bahasa Inggris.
Status izin penggunaan pretrained dari dosen: [isi status aktual].

Gunakan kontrak versi 1 dalam README ini. Jelaskan bentuk tensor dan
alasan keputusan teknis. Jangan mengubah kontrak tanpa menjelaskan
dampaknya terhadap modul anggota lain. Jangan menyalin kode tutorial.
Catat asumsi, dan jangan mengarang hasil eksperimen atau riwayat error.

Mulai dengan merangkum kontrak dan keputusan proyek yang relevan
dengan peran saya. Setelah itu, arahkan saya mengerjakan target sesi
secara bertahap sesuai pembagian tugas. Belum perlu membuat seluruh
kode proyek sekaligus. Jika target saya memerlukan modul anggota lain,
jelaskan dependensinya dan gunakan antarmuka yang telah disepakati.
```

### Identitas sesi

```text
Anggota: [nama dan nomor anggota]
Modul: [data / arsitektur / training / debugging / evaluasi]
Kontrak: versi 1
Target sesi: [...]
File yang dikerjakan: [...]
Keputusan terbaru yang relevan: [...]
```

## Dataset dan preprocessing

- Dataset yang direncanakan: **Flickr8k**.
- Periksa jumlah gambar, pasangan caption, dan berkas yang hilang pada salinan dataset yang dipakai; jangan mengasumsikan semua distribusi memiliki jumlah identik.
- Gunakan split resmi jika tersedia. Jika membuat split sendiri, lakukan **per gambar**, catat pembagiannya, dan gunakan seed 42.
- Seluruh caption dari satu gambar harus berada dalam split yang sama.
- Bangun vocabulary hanya dari caption **training**, dengan ambang frekuensi awal 5.
- Tetapkan panjang caption maksimum 30 token, termasuk token awal dan akhir. Sisakan tempat untuk `<end>` ketika memotong caption.
- Pertahankan seluruh referensi caption yang tersedia untuk evaluasi per gambar.
- Gunakan preprocessing yang sesuai dengan bobot `IMAGENET1K_V2`; samakan preprocessing ekstraksi fitur dan demo gambar baru.
- Ekstrak fitur dalam mode `eval` tanpa gradien. Simpan fitur CPU dengan pemetaan nama gambar yang jelas.

| Token | ID | Fungsi |
|---|---:|---|
| `<pad>` | 0 | Padding |
| `<start>` | 1 | Awal caption |
| `<end>` | 2 | Akhir caption |
| `<unk>` | 3 | Kata di luar vocabulary |

## Kontrak data versi 1

`B` adalah batch size, `T` panjang caption lengkap, dan `V` ukuran vocabulary.

| Komponen | Kontrak |
|---|---|
| `features` | FloatTensor `[B, 2048]` |
| `captions` | LongTensor `[B, 30]` |
| Input decoder | `captions[:, :-1]`, bentuk `[B, 29]` |
| Target | `captions[:, 1:]`, bentuk `[B, 29]` |
| Output model | Logits `[B, 29, V]` |
| State awal LSTM | `h0` dan `c0` masing-masing `[1, B, 512]` |

`forward` mengikuti panjang input token yang diterima; panjang 29 berasal dari baseline caption lengkap 30 token.

## Konfigurasi baseline

| Parameter | Nilai awal |
|---|---|
| Encoder | ResNet50 pretrained `IMAGENET1K_V2`, beku, tanpa classifier |
| Fitur encoder | 2048 dimensi |
| Proyeksi state | Dua linear terpisah `2048 → 512`, untuk `h0` dan `c0` |
| Embedding | 256 dimensi, `padding_idx=0` |
| LSTM | 1 lapis, hidden size 512, `batch_first=True` |
| Dropout | 0.5 pada output LSTM sebelum linear vocabulary |
| Training | Teacher forcing |
| Loss | Cross-entropy, `ignore_index=0` |
| Optimizer | Adam, learning rate awal `1e-3` |
| Parameter optimizer | Seluruh decoder, termasuk embedding, proyeksi state, LSTM, dan output layer |
| Batch size | 64, disesuaikan dengan kapasitas runtime |
| Batas epoch baseline | 15, dengan early stopping |
| Gradient clipping | Norm maksimum 5.0 |
| Seed | 42; catat perangkat dan versi paket karena seed saja tidak menjamin hasil identik |
| Decoding awal | Greedy; beam search sebagai pengembangan |

Untuk menghitung loss, ratakan logits menjadi `[B × (T−1), V]` dan target menjadi `[B × (T−1)]`, atau susun dimensi kelas secara benar. Tidak perlu softmax sebelum cross-entropy.

## Struktur berkas yang direncanakan

```text
kelompok9-image-captioning/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   └── train_utils.py
├── notebooks/
│   ├── 01_data.ipynb
│   ├── 02_model.ipynb
│   ├── 03_train.ipynb
│   └── 04_demo.ipynb
├── docs/
│   └── arsitektur.md
└── logs/
    ├── 01_data.md
    ├── 02_arsitektur.md
    ├── 03_training.md
    ├── 04_debugging.md
    └── 05_eksperimen.md
```

Dataset, fitur, checkpoint, serta hasil besar disimpan di Drive dalam folder `Kelompok9/data`, `features`, `checkpoints`, dan `results`. Jangan memasukkan dataset atau bobot besar ke GitHub. File sementara boleh di penyimpanan lokal Colab, tetapi hasil penting harus disalin ke penyimpanan persisten.

## Urutan implementasi dan menjalankan proyek

Urutan ini menjadi acuan setelah modul dan notebook tersedia.

1. **Persiapan tim:** sepakati kontrak dan status izin pretrained; buat notebook Gemini, repository, dan folder Drive.
2. **`01_data.ipynb`:** siapkan dataset, split, vocabulary, preprocessing, fitur, dan data loader.
3. **`02_model.ipynb`:** bangun decoder; periksa forward dengan data dummy, loss, backward, serta greedy decoding. Bagian ini dapat dimulai sambil menunggu data.
4. **Integrasi:** gunakan satu batch nyata; periksa shape, ID token, device, aliran gradien, serta pemetaan gambar dan caption.
5. **Uji subset kecil:** periksa kemampuan model mempelajari pasangan yang sama sebelum training penuh. Ini pemeriksaan pipeline, bukan bukti generalisasi.
6. **`03_train.ipynb`:** latih baseline, simpan checkpoint terbaik dan log train/validation loss.
7. **Eksperimen:** ubah satu faktor per percobaan dan pilih konfigurasi berdasarkan validation. Perbandingan greedy dan beam menggunakan checkpoint yang sama.
8. **Evaluasi akhir:** kunci konfigurasi dan metode decoding sebelum mengevaluasi test.
9. **`04_demo.ipynb`:** muat checkpoint, vocabulary, konfigurasi, dan encoder yang cocok; tampilkan caption untuk gambar test dan gambar baru.
10. **Finalisasi:** lengkapi hasil, log interaksi, kontribusi anggota, dan laporan.

Checkpoint untuk melanjutkan training memuat bobot model, optimizer/scheduler state jika digunakan, epoch, serta konfigurasi. Demo juga membutuhkan vocabulary, ID token, dan spesifikasi encoder/preprocessing yang cocok.

## Evaluasi dan status hasil

Gunakan BLEU-1 sampai BLEU-4 dengan daftar gambar, preprocessing teks, dan konfigurasi smoothing yang konsisten. Evaluasi setiap gambar sekali dengan seluruh referensi yang tersedia. Hapus token khusus sebelum perhitungan dan tulis skala skor yang dipakai.

Lengkapi metrik dengan contoh berhasil dan gagal. Nilai juga kesesuaian objek, aktivitas, dan hubungan dalam caption dengan isi gambar.

**Belum ada hasil training atau metrik yang dicantumkan pada README awal ini.**

| Eksperimen | Perubahan | BLEU-1 validation | BLEU-4 validation |
|---|---|---|---|
| Baseline | Konfigurasi awal + greedy | Belum diukur | Belum diukur |
| Tuning | Satu faktor per eksperimen | Belum diukur | Belum diukur |
| Decoding | Greedy vs beam 3/5, checkpoint sama | Belum diukur | Belum diukur |

## Log interaksi dan kontribusi

Setiap sesi mencatat nama anggota, waktu WIB, nama model LLM yang benar-benar digunakan, target modul, prompt, alasan, ringkasan respons, pemeriksaan aktual, serta share link atau ekspor. Jika ada error, simpan traceback, analisis, dan hasil perbaikannya.

Jangan membuat hasil training atau error fiktif. Catat keputusan yang diterima dan bagian jawaban LLM yang diubah atau tidak dipakai. Setiap anggota menyimpan kontribusi kode dan dokumentasinya melalui commit yang jelas.

## Acuan

- Panduan Proyek 1 Vibe Coding dari dosen.
- Workflow Kelompok 9 — Image Captioning (CNN + LSTM), 30 Oktober 2026.
- Panduan Kerja Anggota 1 Kelompok 9.
- [Notebooks di Gemini Apps](https://support.google.com/gemini/answer/16972047?hl=en).
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html).
- [Dokumentasi ResNet50 Torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html).
