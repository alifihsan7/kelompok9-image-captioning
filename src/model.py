"""
src/model.py
Modul arsitektur Decoder LSTM untuk Image Captioning.
Mengikuti spesifikasi Kontrak Teknis Data V1 (Kelompok 9).
"""

from typing import List, Dict, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        if feature.dim() == 1:
            if feature.shape[0] != self.feature_dim:
                raise ValueError(
                    f"Fitur gambar 1D harus berukuran [{self.feature_dim}], "
                    f"tetapi diterima bentuk {list(feature.shape)}."
                )
            feature = feature.unsqueeze(0)
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

        target_device = next(self.parameters()).device
        if feature.device != target_device:
            feature = feature.to(target_device)

        was_training = self.training
        self.eval()

        generated_tokens: List[int] = []

        try:
            with torch.no_grad():
                h = self.init_h(feature).unsqueeze(0)
                c = self.init_c(feature).unsqueeze(0)

                current_token = torch.tensor(
                    [[start_token_id]], dtype=torch.long, device=target_device
                )

                for _ in range(max_len):
                    emb = self.embedding(current_token)
                    lstm_out, (h, c) = self.lstm(emb, (h, c))
                    logits = self.fc_out(lstm_out.squeeze(1))

                    logits[0, pad_token_id] = -float("inf")
                    logits[0, start_token_id] = -float("inf")

                    pred_token = torch.argmax(logits, dim=-1).item()

                    if pred_token == end_token_id:
                        break

                    generated_tokens.append(pred_token)
                    current_token = torch.tensor(
                        [[pred_token]], dtype=torch.long, device=target_device
                    )
        finally:
            self.train(was_training)

        return generated_tokens

    def generate_beam(
        self,
        feature: torch.Tensor,
        beam_size: int = 3,
        max_len: int = 30,
        alpha: float = 0.7,
        start_token_id: int = 1,
        end_token_id: int = 2,
        pad_token_id: int = 0,
    ) -> List[int]:
        """
        Membangkitkan deskripsi gambar menggunakan pencarian beam (Beam Search).

        Mekanisme:
            - Memelihara satu kumpulan beam dengan kapasitas maksimal beam_size kandidat (aktif dan selesai).
            - Mengakumulasi log-probabilitas via log_softmax pada setiap langkah waktu.
            - Status LSTM (h, c) memiliki definisi waktu konsisten: status awal h0/c0 diproyeksikan dari gambar,
              seluruh cabang anak dari induk yang sama berbagi h/c hasil langkah induk, dan status baru bercabang
              setelah token masing-masing diproses tanpa modifikasi in-place.
            - Token PAD (0) dan START (1) dimask ke -inf setelah log_softmax, dan skor non-finite dibuang.
            - Pemilihan top-k ekspansi menggunakan stable argsort sehingga jika skor identik, token dengan ID
              lebih kecil diprioritaskan sebelum pemotongan k_safe.
            - Kandidat selesai ditandai saat memprediksi token END (2). Token END ikut dihitung ke dalam skor
              dan panjang L, tetapi tidak dimasukkan ke dalam daftar kembalian tokens.
            - Normalisasi panjang tunggal: score = cumulative_log_prob / (((5 + L) / 6) ** alpha).
            - Pemilihan akhir diurutkan secara deterministik berdasarkan skor normalisasi tertinggi.

        Args:
            feature (torch.Tensor): Fitur visual citra tunggal berdimensi [feature_dim] atau [1, feature_dim].
            beam_size (int, optional): Jumlah kandidat paralel yang dipertahankan. Default: 3.
            max_len (int, optional): Batas maksimal langkah iterasi pembangkitan kata. Default: 30.
            alpha (float, optional): Eksponen penalti panjang sekuens. Default: 0.7.
            start_token_id (int, optional): ID token khusus <start>. Default: 1.
            end_token_id (int, optional): ID token khusus <end>. Default: 2.
            pad_token_id (int, optional): ID token khusus <pad>. Default: 0.

        Returns:
            List[int]: Daftar indeks kata hasil inferensi (tanpa token <start>, <end>, maupun <pad>).

        Raises:
            ValueError: Jika bentuk feature tidak valid atau beam_size < 1.
            RuntimeError: Jika tidak ada kandidat valid yang tersisa dalam beam.
        """
        # 1. Validasi bentuk tensor input
        if feature.dim() == 1:
            if feature.shape[0] != self.feature_dim:
                raise ValueError(
                    f"Fitur gambar 1D harus berukuran [{self.feature_dim}], "
                    f"tetapi diterima bentuk {list(feature.shape)}."
                )
            feature = feature.unsqueeze(0)
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

        if beam_size < 1:
            raise ValueError(f"beam_size harus >= 1, tetapi diterima {beam_size}.")

        # 2. Penyelarasan device antara bobot model dan tensor operasional
        target_device = next(self.parameters()).device
        if feature.device != target_device:
            feature = feature.to(target_device)

        # 3. Manajemen mode model via try/finally
        was_training = self.training
        self.eval()

        try:
            with torch.no_grad():
                h0 = self.init_h(feature).unsqueeze(0)
                c0 = self.init_c(feature).unsqueeze(0)

                # Kandidat awal: START diproses pada langkah pertama
                # START tidak dihitung dalam L (L = 0)
                initial_candidate: Dict[str, Any] = {
                    "tokens": [],
                    "cum_log_prob": 0.0,
                    "score": 0.0,
                    "h": h0,
                    "c": c0,
                    "is_done": False,
                    "length": 0,
                }
                beam: List[Dict[str, Any]] = [initial_candidate]

                for _ in range(max_len):
                    all_candidates: List[Dict[str, Any]] = []

                    for cand in beam:
                        if cand["is_done"]:
                            # Pertahankan kandidat selesai tanpa diekspansi lagi dan tanpa mengubah skor
                            all_candidates.append(cand)
                            continue

                        # Tentukan input token untuk langkah aktif
                        if len(cand["tokens"]) == 0:
                            input_token_id = start_token_id
                        else:
                            input_token_id = cand["tokens"][-1]

                        current_token = torch.tensor(
                            [[input_token_id]], dtype=torch.long, device=target_device
                        )

                        emb = self.embedding(current_token)
                        lstm_out, (next_h, next_c) = self.lstm(emb, (cand["h"], cand["c"]))
                        logits = self.fc_out(lstm_out.squeeze(1))

                        # Log-softmax untuk akumulasi probabilitas numerik stabil
                        log_probs = F.log_softmax(logits, dim=-1).squeeze(0)

                        # Masking PAD dan START menjadi -inf
                        log_probs[pad_token_id] = -float("inf")
                        log_probs[start_token_id] = -float("inf")

                        # Ambil hanya token dengan nilai finite (membuang PAD, START, dan skor non-finite)
                        finite_mask = torch.isfinite(log_probs)
                        finite_indices = torch.nonzero(finite_mask, as_tuple=False).squeeze(-1)

                        if finite_indices.numel() > 0:
                            finite_log_probs = log_probs[finite_indices]
                            # Pengurutan stabil descending: jika skor sama, token ID lebih kecil diprioritaskan
                            sort_perm = torch.argsort(finite_log_probs, descending=True, stable=True)
                            k_safe = min(beam_size, sort_perm.numel())
                            selected_indices = finite_indices[sort_perm[:k_safe]].tolist()
                            selected_vals = finite_log_probs[sort_perm[:k_safe]].tolist()

                            for val, idx in zip(selected_vals, selected_indices):
                                new_cum = cand["cum_log_prob"] + val
                                new_len = cand["length"] + 1

                                # Rumus normalisasi panjang tunggal
                                length_penalty = ((5.0 + new_len) / 6.0) ** alpha
                                new_score = new_cum / length_penalty

                                if idx == end_token_id:
                                    # END dihitung dalam skor dan panjang L, tetapi dibuang dari list tokens
                                    all_candidates.append({
                                        "tokens": list(cand["tokens"]),
                                        "cum_log_prob": new_cum,
                                        "score": new_score,
                                        "h": next_h,
                                        "c": next_c,
                                        "is_done": True,
                                        "length": new_len,
                                    })
                                else:
                                    all_candidates.append({
                                        "tokens": cand["tokens"] + [idx],
                                        "cum_log_prob": new_cum,
                                        "score": new_score,
                                        "h": next_h,
                                        "c": next_c,
                                        "is_done": False,
                                        "length": new_len,
                                    })

                    # Validasi ketersediaan kandidat valid
                    if len(all_candidates) == 0:
                        raise RuntimeError("Tidak ada kandidat valid yang tersisa dalam beam search.")

                    # Seleksi global K terbaik dengan tie-breaker deterministik
                    # (-score, tokens, -is_done)
                    all_candidates.sort(key=lambda x: (-x["score"], x["tokens"], -int(x["is_done"])))
                    beam = all_candidates[:beam_size]

                    # Hentikan jika seluruh kandidat yang benar-benar tersimpan telah selesai
                    if len(beam) > 0 and all(c["is_done"] for c in beam):
                        break

                if len(beam) == 0:
                    raise RuntimeError("Tidak ada kandidat valid yang tersisa dalam beam search.")

                # Pilih kandidat dengan skor ternormalisasi tertinggi di antara semua kandidat tersisa
                beam.sort(key=lambda x: (-x["score"], x["tokens"], -int(x["is_done"])))
                best_candidate = beam[0]
                return best_candidate["tokens"]
        finally:
            self.train(was_training)
