"""
src/experiment.py
Modul otomatisasi eksperimen tuning hiperparameter & data augmentation Image Captioning (Kelompok 9).

Menyediakan fungsi:
1. run_experiment: Menjalankan 1 alur eksperimen penuh end-to-end
   (seed -> loaders [use_flip] -> model -> fit [weight_decay, label_smoothing] -> evaluate BLEU -> plot -> log CSV & JSON).
2. summarize_experiments: Menampilkan tabel ringkasan experiments.csv diurutkan dari BLEU-4 tertinggi.
"""

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import torch

from model import CaptionDecoder
from dataset import get_loaders, get_eval_set
from train_utils import set_seed, fit
from eval_utils import evaluate_bleu, plot_history


def run_experiment(
    config: Dict[str, Any],
    root: Union[str, Path],
    feat_dir: Union[str, Path],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Menjalankan satu siklus eksperimen penuh secara deterministik dan mencatat
    hasilnya ke tabel experiments.csv serta berkas JSON.

    Args:
        config (Dict[str, Any]): Konfigurasi eksperimen (exp_name, epochs, lr, batch_size,
            embed_dim, hidden_size, dropout_p, use_flip, weight_decay, label_smoothing,
            lr_patience, early_stopping_patience, lr_factor, seed).
        root (Union[str, Path]): Jalur direktori utama Google Drive proyek Kelompok 9.
        feat_dir (Union[str, Path]): Jalur folder penyimpanan fitur ResNet-50 dan kosakata.
        device (torch.device): Perangkat komputasi GPU/CPU.

    Returns:
        Dict[str, Any]: Ringkasan hasil pelatihan dan evaluasi metrik BLEU.
    """
    start_total_time = time.time()
    root_path = Path(root)
    results_dir = root_path / "results"
    checkpoints_dir = root_path / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Ekstraksi konfigurasi
    exp_name = config["exp_name"]
    epochs = config.get("epochs", 15)
    lr = config.get("lr", 1e-3)
    batch_size = config.get("batch_size", 64)
    embed_dim = config.get("embed_dim", 256)
    hidden_size = config.get("hidden_size", 512)
    dropout_p = config.get("dropout_p", 0.5)
    use_flip = bool(config.get("use_flip", False))
    weight_decay = float(config.get("weight_decay", 0.0))
    label_smoothing = float(config.get("label_smoothing", 0.0))
    lr_patience = config.get("lr_patience", 1)
    early_stopping_patience = config.get("early_stopping_patience", 4)
    lr_factor = config.get("lr_factor", 0.5)
    seed = config.get("seed", 42)

    print(f"\n{'='*80}")
    print(f"MEMULAI EKSPERIMEN: {exp_name}")
    print(f"Config: {config}")
    print(f"{'='*80}")

    # 1. Kunci seed acak
    set_seed(seed)

    # 2. Muat DataLoader dan normalisasi kosakata
    train_loader, val_loader, _, vocab = get_loaders(
        batch_size=batch_size,
        features_dir=feat_dir,
        use_flip=use_flip,
    )
    word2idx = vocab["word2idx"]
    idx2word = vocab["idx2word"]
    if isinstance(idx2word, list):
        idx2word = {i: w for i, w in enumerate(idx2word)}
        vocab["idx2word"] = idx2word
    vocab_size = len(word2idx)

    # 3. Inisialisasi model CaptionDecoder
    model = CaptionDecoder(
        vocab_size=vocab_size,
        feature_dim=2048,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        dropout_p=dropout_p,
        pad_idx=0,
    ).to(device)

    # 4. Pelatihan model via fit()
    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        exp_name=exp_name,
        root=root,
        lr_patience=lr_patience,
        early_stopping_patience=early_stopping_patience,
        lr_factor=lr_factor,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
    )

    # 5. Muat checkpoint terbaik
    checkpoint_file = checkpoints_dir / f"best_{exp_name}.pt"
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_epoch = checkpoint.get("epoch")
    best_val_loss = checkpoint.get("val_loss")
    best_train_loss = checkpoint.get("train_loss")
    stopped_epoch = history.get("stopped_epoch", len(history.get("epoch", [])))

    print(f"\nMemuat bobot terbaik: Epoch {best_epoch} (Val Loss: {best_val_loss:.4f})")

    # 6. Muat subset evaluasi validasi
    _, feature_tensor_val, references_val = get_eval_set("val", features_dir=feat_dir)
    eval_items_val = list(zip(feature_tensor_val, references_val))

    # 7. Evaluasi metrik BLEU-1 sampai BLEU-4
    print("Menghitung skor korpus BLEU pada subset validasi...")
    bleu_scores = evaluate_bleu(
        model=model,
        eval_items=eval_items_val,
        idx2word=idx2word,
        device=device,
        max_len=30,
    )

    # 8. Visualisasi kurva loss
    csv_history_path = results_dir / f"{exp_name}.csv"
    loss_plot_path = results_dir / f"{exp_name}_loss.png"
    if csv_history_path.is_file():
        plot_history(csv_path=csv_history_path, save_path=loss_plot_path)

    total_seconds = round(time.time() - start_total_time, 2)

    # 9. Simpan berkas ringkasan JSON
    summary = {
        "exp_name": exp_name,
        "config": config,
        "best_epoch": best_epoch,
        "best_val_loss": round(float(best_val_loss), 4) if best_val_loss is not None else None,
        "best_train_loss": round(float(best_train_loss), 4) if best_train_loss is not None else None,
        "stopped_epoch": stopped_epoch,
        "bleu_scores": {k: round(float(v), 4) for k, v in bleu_scores.items()},
        "total_seconds": total_seconds,
    }
    json_path = results_dir / f"{exp_name}_bleu.json"
    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    # 10. Tambah/Perbarui baris pada experiments.csv
    csv_experiments = results_dir / "experiments.csv"
    row_data = {
        "exp_name": exp_name,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "embed_dim": embed_dim,
        "hidden_size": hidden_size,
        "dropout_p": dropout_p,
        "use_flip": use_flip,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "lr_patience": str(lr_patience) if lr_patience is not None else "None",
        "early_stopping_patience": str(early_stopping_patience) if early_stopping_patience is not None else "None",
        "lr_factor": lr_factor,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": round(float(best_val_loss), 4) if best_val_loss is not None else None,
        "stopped_epoch": stopped_epoch,
        "BLEU-1": round(float(bleu_scores["BLEU-1"]), 4),
        "BLEU-2": round(float(bleu_scores["BLEU-2"]), 4),
        "BLEU-3": round(float(bleu_scores["BLEU-3"]), 4),
        "BLEU-4": round(float(bleu_scores["BLEU-4"]), 4),
        "total_seconds": total_seconds,
    }

    if csv_experiments.is_file():
        df_exp = pd.read_csv(csv_experiments)
        # Hapus entri lama jika exp_name sama
        df_exp = df_exp[df_exp["exp_name"].astype(str) != str(exp_name)]
        df_exp = pd.concat([df_exp, pd.DataFrame([row_data])], ignore_index=True)
    else:
        df_exp = pd.DataFrame([row_data])

    df_exp.to_csv(csv_experiments, index=False)
    print(f"Catatan eksperimen berhasil disimpan ke: {csv_experiments}")

    return summary


def summarize_experiments(root: Union[str, Path] = ".") -> pd.DataFrame:
    """
    Membaca berkas experiments.csv dan mengembalikan DataFrame pandas
    yang diurutkan berdasarkan skor BLEU-4 tertinggi.
    """
    root_path = Path(root)
    csv_experiments = root_path / "results" / "experiments.csv"
    if not csv_experiments.is_file():
        print(f"Peringatan: Berkas {csv_experiments} belum ditemukan.")
        return pd.DataFrame()

    df = pd.read_csv(csv_experiments)
    if "BLEU-4" in df.columns:
        df = df.sort_values(by="BLEU-4", ascending=False).reset_index(drop=True)
    return df
