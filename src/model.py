"""
src/model.py
Modul arsitektur Decoder LSTM untuk Image Captioning.
Mengikuti spesifikasi Kontrak Teknis Data V1 (Kelompok 9).
"""

from typing import List
import torch
import torch.nn as nn


class CaptionDecoder(nn.Module):
    """
    Decoder LSTM untuk Image Captioning.

    Arsitektur:
        - Proyeksi linear fitur visual [B, feature_dim] ke status awal LSTM (h0, c0).
        - Lapisan embedding token kata dengan penanganan padding_idx.
        - LSTM 1 lapis (batch_first=True) untuk pemrosesan sekuens teks.
        - Regulerisasi Dropout pada representasi status tersembunyi.
        - Lapisan linear proyeksi akhir ke dimensi kosakata (raw logits).

    Args:
        vocab_size (int): Jumlah total kosakata unik (V).
        feature_dim (int, optional): Dimensi vektor fitur visual encoder ResNet-50. Default: 2048.
        embed_dim (int, optional): Dimensi representasi vektor kata. Default: 256.
        hidden_size (int, optional): Dimensi ruang status tersembunyi (hidden & cell state). Default: 512.
        dropout_p (float, optional): Probabilitas dropout regulerisasi. Default: 0.5.
        pad_idx (int, optional): Indeks token khusus <pad>. Default: 0.
    """

    def __init__(
        self,
        vocab_size: int,
        feature_dim: int = 2048,
        embed_dim: int = 256,
        hidden_size: int = 512,
        dropout_p: float = 0.5,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.pad_idx = pad_idx

        # 1. Proyeksi fitur visual ke hidden state (h0) dan cell state (c0) awal
        # [B, feature_dim] -> [B, hidden_size]
        self.init_h = nn.Linear(feature_dim, hidden_size)
        self.init_c = nn.Linear(feature_dim, hidden_size)

        # 2. Lapisan penyematan kata (word embedding)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        # 3. Lapisan recurrent LSTM 1 lapis
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # 4. Lapisan regulerisasi dropout
        self.dropout = nn.Dropout(p=dropout_p)

        # 5. Lapisan linear keluaran menuju ruang kosakata
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(
        self, features: torch.Tensor, captions_in: torch.Tensor
    ) -> torch.Tensor:
        """
        Menjalankan satu lintasan komputasi maju menggunakan skema Teacher Forcing.

        Bentuk Tensor:
            - features:    torch.FloatTensor [B, feature_dim]
            - captions_in: torch.LongTensor  [B, L]
            - logits:      torch.FloatTensor [B, L, vocab_size]

        Args:
            features (torch.Tensor): Representasi visual dari encoder berdimensi [B, feature_dim].
            captions_in (torch.Tensor): Sekuens token masukan berdimensi [B, L].

        Returns:
            torch.Tensor: Logits (skor mentah sebelum aktivasi softmax) berdimensi [B, L, vocab_size].
        """
        # 1. Inisialisasi h0 dan c0 dari fitur visual
        # [B, feature_dim] -> [B, hidden_size] -> [1, B, hidden_size]
        h0 = self.init_h(features).unsqueeze(0)
        c0 = self.init_c(features).unsqueeze(0)

        # 2. Pemetaan token teks ke ruang kontinu
        # [B, L] -> [B, L, embed_dim]
        embedded = self.embedding(captions_in)

        # 3. Propagasi sekuensial LSTM
        # lstm_out berdimensi [B, L, hidden_size]
        lstm_out, _ = self.lstm(embedded, (h0, c0))

        # 4. Penerapan dropout pada luaran hidden state
        out = self.dropout(lstm_out)

        # 5. Proyeksi skor mentah ke kosakata tanpa softmax
        # [B, L, hidden_size] -> [B, L, vocab_size]
        logits = self.fc_out(out)

        return logits

    def generate_greedy(
        self,
        feature: torch.Tensor,
        max_len: int = 30,
        start_token_id: int = 1,
        end_token_id: int = 2,
        pad_token_id: int = 0,
    ) -> List[int]:
        """
        Membangkitkan token deskripsi gambar secara greedy langkah demi langkah (autoregresif).

        Perbedaan Batas Panjang:
            - max_len=30 di sini menentukan batas maksimal langkah inferensi (maksimal 30 token prediksi).
            - Ini berbeda dari panjang sekuens training T=30 yang mencakup <start>, kata isi, <end>, dan <pad>.

        Bentuk Tensor Masukan:
            - feature: [feature_dim] (1D) atau [1, feature_dim] (2D)

        Args:
            feature (torch.Tensor): Fitur visual citra tunggal berdimensi [feature_dim] atau [1, feature_dim].
            max_len (int, optional): Batas maksimal langkah prediksi kata. Default: 30.
            start_token_id (int, optional): ID token khusus <start>. Default: 1.
            end_token_id (int, optional): ID token khusus <end>. Default: 2.
            pad_token_id (int, optional): ID token khusus <pad>. Default: 0.

        Returns:
            List[int]: Daftar indeks kata hasil inferensi (tanpa token <start>, <end>, maupun <pad>).

        Raises:
            ValueError: Jika bentuk tensor fitur tidak sesuai dengan dimensi [feature_dim] atau [1, feature_dim].
        """
        # 1. Validasi bentuk tensor input secara ketat
        if feature.dim() == 1:
            if feature.shape[0] != self.feature_dim:
                raise ValueError(
                    f"Fitur gambar 1D harus berukuran [{self.feature_dim}], "
                    f"tetapi diterima bentuk {list(feature.shape)}."
                )
            feature = feature.unsqueeze(0)  # Menjadi [1, feature_dim]
        elif feature.dim() == 2:
            if feature.shape != (1, self.feature_dim):
                raise ValueError(
                    f"Fitur gambar 2D harus berukuran [1, {self.feature_dim}], "
                    f"tetapi diterima bentuk {list(feature.shape)}."
                )
        else:
            raise ValueError(
                f"Fitur gambar harus berdimensi 1D [{self.feature_dim}] atau 2D [1, {self.feature_dim}], "
                f"tetapi diterima tensor {feature.dim()}D dengan bentuk {list(feature.shape)}."
            )

        # 2. Penyelarasan device antara model, fitur, dan tensor token
        target_device = next(self.parameters()).device
        if feature.device != target_device:
            feature = feature.to(target_device)

        # 3. Manajemen status mode pelatihan (eval & restore via try/finally)
        was_training = self.training
        self.eval()

        generated_tokens: List[int] = []

        try:
            with torch.no_grad():
                # Inisialisasi status tersembunyi awal dari fitur gambar
                # [1, feature_dim] -> [1, hidden_size] -> [1, 1, hidden_size]
                h = self.init_h(feature).unsqueeze(0)
                c = self.init_c(feature).unsqueeze(0)

                # Token pemantik awal: <start> berdimensi [1, 1]
                current_token = torch.tensor(
                    [[start_token_id]], dtype=torch.long, device=target_device
                )

                # Iterasi prediksi autoregresif hingga max_len langkah
                for _ in range(max_len):
                    # Embedding token saat ini: [1, 1] -> [1, 1, embed_dim]
                    emb = self.embedding(current_token)

                    # Propagasi recurrent 1 langkah waktu, perbarui h dan c
                    lstm_out, (h, c) = self.lstm(emb, (h, c))

                    # Proyeksi ke skor mentah kosakata: [1, vocab_size]
                    logits = self.fc_out(lstm_out.squeeze(1))

                    # Masking token PAD dan START agar tidak pernah terpilih
                    logits[0, pad_token_id] = -float("inf")
                    logits[0, start_token_id] = -float("inf")

                    # Pemilihan kata greedy (skor tertinggi)
                    pred_token = torch.argmax(logits, dim=-1).item()

                    # Hentikan jika mencapai penutup <end> tanpa memasukkannya ke hasil
                    if pred_token == end_token_id:
                        break

                    generated_tokens.append(pred_token)

                    # Siapkan input untuk langkah waktu berikutnya
                    current_token = torch.tensor(
                        [[pred_token]], dtype=torch.long, device=target_device
                    )
        finally:
            # Pulihkan status pelatihan awal
            self.train(was_training)

        return generated_tokens
