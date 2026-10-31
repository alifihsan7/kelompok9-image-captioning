"""
src/model.py
Modul arsitektur Decoder LSTM untuk Image Captioning.
Mengikuti spesifikasi Kontrak Teknis Data V1 (Kelompok 9).
"""

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
