"""
src/eval_utils.py
Modul evaluasi metrik dan visualisasi inferensi Image Captioning (Kelompok 9).

Pembaruan V2:
- evaluate_bleu dan show_samples memanggil model.generate_greedy langsung per gambar
  (feature [2048]) sesuai antarmuka resmi model.py.
- Fallback try/except batch dan parameter batch_size dihapus.
- Progress bar tqdm disematkan untuk pemantauan evaluasi dataset validasi.
- Backend matplotlib Agg dihapus agar visualisasi grafik interaktif di notebook tetap optimal.
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import nltk
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
import torch
import torch.nn as nn
from tqdm.auto import tqdm


def decode_tokens(
    ids: Union[torch.Tensor, List[int]],
    idx2word: Dict[int, str],
) -> List[str]:
    """
    Mengonversi sekuens ID token menjadi deretan kata teks (list of words).

    Aturan Kontrak Teknis Data V1:
    - Berhenti seketika saat token <end> (ID 2) ditemukan.
    - Mengabaikan token <start> (ID 1) dan <pad> (ID 0).
    - Memetakan token lainnya ke kata aslinya via kamus idx2word.
    """
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()

    if len(ids) > 0 and isinstance(ids[0], list):
        ids = ids[0]

    words: List[str] = []
    for token_id in ids:
        token_id = int(token_id)
        if token_id == 2:  # Token <end>
            break
        if token_id in (0, 1):  # Abaikan <pad> dan <start>
            continue
        words.append(idx2word.get(token_id, "<unk>"))

    return words


def evaluate_bleu(
    model: nn.Module,
    eval_items: List[Tuple[torch.Tensor, List[List[str]]]],
    idx2word: Dict[int, str],
    device: torch.device,
    max_len: int = 30,
) -> Dict[str, float]:
    """
    Menghitung skor korpus BLEU-1 sampai BLEU-4 terhadap kumpulan evaluasi
    dengan memproses inferensi greedy langsung per citra tunggal.

    Args:
        model (nn.Module): Model decoder image captioning dengan method generate_greedy(feature, max_len).
        eval_items (List[Tuple[torch.Tensor, List[List[str]]]]): Pasangan (fitur [2048], 5 caption referensi).
        idx2word (Dict[int, str]): Kamus pemetaan ID token integer ke kata teks.
        device (torch.device): Perangkat komputasi aktif (GPU/CPU).
        max_len (int, optional): Batas maksimum langkah inferensi greedy. Default: 30.

    Returns:
        Dict[str, float]: Skor metrik BLEU-1, BLEU-2, BLEU-3, dan BLEU-4.
    """
    model.eval()
    references: List[List[List[str]]] = []
    hypotheses: List[List[str]] = []

    with torch.no_grad():
        for feature, ref_list in tqdm(eval_items, desc="Evaluasi BLEU"):
            references.append(ref_list)
            # Pemanggilan langsung per citra tunggal [2048]
            pred_ids = model.generate_greedy(feature.to(device), max_len=max_len)
            hypotheses.append(decode_tokens(pred_ids, idx2word))

    smooth = SmoothingFunction().method1
    bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=smooth)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0.0, 0.0), smoothing_function=smooth)
    bleu3 = corpus_bleu(references, hypotheses, weights=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0), smoothing_function=smooth)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    return {
        "BLEU-1": float(bleu1),
        "BLEU-2": float(bleu2),
        "BLEU-3": float(bleu3),
        "BLEU-4": float(bleu4),
    }


def plot_history(
    csv_path: Union[str, Path],
    save_path: Union[str, Path],
) -> None:
    """
    Membuat grafik garis train_loss vs val_loss per epoch dari berkas CSV log hasil fit()
    dan menyimpannya sebagai file gambar PNG.
    """
    csv_path = Path(csv_path)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs: List[int] = []
    train_losses: List[float] = []
    val_losses: List[float] = []

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", color="#1f77b4", linewidth=2, label="Train Loss")
    plt.plot(epochs, val_losses, marker="s", color="#d62728", linewidth=2, label="Val Loss")

    plt.title(f"Kurva Loss Pelatihan ({csv_path.stem})", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss (CrossEntropy)", fontsize=11)
    if len(epochs) > 0:
        plt.xticks(epochs)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()


def show_samples(
    model: nn.Module,
    eval_items: List[Tuple[torch.Tensor, List[List[str]]]],
    idx2word: Dict[int, str],
    device: torch.device,
    n: int = 10,
) -> None:
    """
    Mencetak sampel hasil prediksi caption model berdampingan dengan 1 caption referensi.
    """
    model.eval()
    num_samples = min(n, len(eval_items))

    print(f"\n{'='*78}")
    print(f"HASIL PREDIKSI MODEL VS REFERENSI ({num_samples} SAMPEL)")
    print(f"{'='*78}")

    with torch.no_grad():
        for i in range(num_samples):
            feature, references = eval_items[i]
            # Pemanggilan langsung per citra tunggal [2048]
            pred_ids = model.generate_greedy(feature.to(device), max_len=30)

            pred_words = decode_tokens(pred_ids, idx2word)
            pred_text = " ".join(pred_words) if pred_words else "<empty>"

            ref_words = references[0] if len(references) > 0 else []
            ref_text = " ".join(ref_words) if ref_words else "<empty>"

            print(f"[{i+1:02d}]")
            print(f"  Prediksi : {pred_text}")
            print(f"  Referensi: {ref_text}")
            print("-" * 78)
