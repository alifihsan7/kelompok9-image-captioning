"""
src/train_utils.py
Modul pelatihan dan evaluasi PyTorch untuk Image Captioning (Kelompok 9).

Pembaruan V4:
- Menggunakan optimizer AdamW dengan parameter weight_decay (decoupled weight decay).
  Saat weight_decay=0.0, pembaruan bobot identik dengan Adam murni.
- Menambahkan parameter label_smoothing pada CrossEntropyLoss khusus untuk TRAINING.
- Evaluasi validasi (evaluate) tetap menggunakan CrossEntropyLoss(ignore_index=0) murni
  tanpa label smoothing agar nilai val_loss tetap sebanding lintas eksperimen.
- Mempertahankan in-memory CPU caching untuk checkpoint terbaik, penulisan tunggal ke SSD lokal,
  pembersihan file Drive lama, penyalinan via shutil, dan verifikasi torch.load.
"""

import os
import csv
import time
import shutil
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


def set_seed(seed: int = 42) -> None:
    """Mengunci seluruh generator bilangan acak untuk menjamin reproduksibilitas."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    clip: float = 5.0,
) -> float:
    """Melatih model satu epoch penuh dengan skema Teacher Forcing."""
    model.train()
    total_loss = 0.0
    num_batches = len(loader)

    for features, captions in loader:
        features = features.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)

        # Penjajaran Teacher Forcing sesuai kontrak: input [B, 29], target [B, 29]
        captions_in = captions[:, :-1]
        targets = captions[:, 1:]

        optimizer.zero_grad()
        logits = model(features, captions_in)

        vocab_size = logits.size(-1)
        loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Mengevaluasi loss model pada data validasi tanpa gradien."""
    model.eval()
    total_loss = 0.0
    num_batches = len(loader)

    with torch.no_grad():
        for features, captions in loader:
            features = features.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)

            captions_in = captions[:, :-1]
            targets = captions[:, 1:]

            logits = model(features, captions_in)
            vocab_size = logits.size(-1)
            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))

            total_loss += loss.item()

    return total_loss / num_batches if num_batches > 0 else 0.0


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    exp_name: str,
    root: Union[str, Path] = ".",
    lr_patience: Optional[int] = 1,
    early_stopping_patience: Optional[int] = 4,
    lr_factor: float = 0.5,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
) -> Dict[str, Any]:
    """
    Loop pelatihan multi-epoch dengan optimizer AdamW, scheduler ReduceLROnPlateau,
    dan Early Stopping. Checkpoint terbaik dicatat di RAM CPU dan disimpan satu kali di akhir.

    Args:
        model (nn.Module): Model decoder captioning.
        train_loader (DataLoader): DataLoader data latih.
        val_loader (DataLoader): DataLoader data validasi.
        epochs (int): Jumlah maksimum epoch.
        lr (float): Learning rate awal.
        device (torch.device): Perangkat komputasi GPU/CPU.
        exp_name (str): Nama eksperimen.
        root (Union[str, Path], optional): Direktori root Google Drive. Default: '.'.
        lr_patience (Optional[int], optional): Kesabaran penurunan LR. Default: 1.
        early_stopping_patience (Optional[int], optional): Kesabaran Early Stopping. Default: 4.
        lr_factor (float, optional): Faktor pengali penurunan LR. Default: 0.5.
        weight_decay (float, optional): Koefisien weight decay untuk AdamW. Default: 0.0.
        label_smoothing (float, optional): Epsilon label smoothing khusus data latih. Default: 0.0.

    Returns:
        Dict[str, Any]: Kamus riwayat metrik pelatihan dan status epoch henti.
    """
    root_path = Path(root)
    checkpoints_dir = root_path / "checkpoints"
    results_dir = root_path / "results"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    local_checkpoints_dir = Path("/content/checkpoints")
    local_checkpoints_dir.mkdir(parents=True, exist_ok=True)

    local_checkpoint_file = local_checkpoints_dir / f"best_{exp_name}.pt"
    drive_checkpoint_file = checkpoints_dir / f"best_{exp_name}.pt"
    csv_file = results_dir / f"{exp_name}.csv"

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # Loss untuk pelatihan (dengan opsi label smoothing)
    train_criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)
    # Loss untuk validasi tetap murni tanpa label smoothing agar metrik konsisten
    val_criterion = nn.CrossEntropyLoss(ignore_index=0)

    scheduler = None
    if lr_patience is not None:
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_factor, patience=lr_patience, threshold=0.0
        )

    best_val_loss = float("inf")
    best_epoch = -1
    best_train_loss = 0.0
    best_lr = lr
    best_model_state: Optional[Dict[str, torch.Tensor]] = None

    early_stop_counter = 0
    stopped_epoch = epochs

    history: Dict[str, Any] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "seconds": [],
        "stopped_epoch": None,
    }

    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "seconds"])

    print(f"Mulai eksperimen: {exp_name} | Max Epochs: {epochs} | LR Awal: {lr} | Target Device: {device}")
    print(f"Konfigurasi Regularisasi: Weight Decay={weight_decay} | Label Smoothing={label_smoothing}")
    print(f"Fitur: LR Scheduler={f'Aktif (patience={lr_patience}, factor={lr_factor})' if lr_patience is not None else 'Nonaktif'} | "
          f"Early Stopping={f'Aktif (patience={early_stopping_patience})' if early_stopping_patience is not None else 'Nonaktif'}")
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=train_criterion,
            optimizer=optimizer,
            device=device,
            clip=5.0,
        )

        val_loss = evaluate(
            model=model,
            loader=val_loader,
            criterion=val_criterion,
            device=device,
        )

        elapsed_seconds = round(time.time() - start_time, 2)
        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)
        history["seconds"].append(elapsed_seconds)

        with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{current_lr:.6f}", elapsed_seconds])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_train_loss = train_loss
            best_lr = current_lr
            best_model_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            early_stop_counter = 0
            status_simpan = " [BEST MEMORY]"
        else:
            early_stop_counter += 1
            status_simpan = ""

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Waktu: {elapsed_seconds:>5.2f}s{status_simpan}"
        )

        if scheduler is not None:
            old_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < old_lr:
                print(f"  --> [LR REDUCED] Learning rate diturunkan: {old_lr:.6f} -> {new_lr:.6f}")

        if early_stopping_patience is not None and early_stop_counter >= early_stopping_patience:
            print(
                f"  --> [EARLY STOPPING] Pelatihan dihentikan pada Epoch {epoch}. "
                f"val_loss tidak membaik selama {early_stopping_patience} epoch berturut-turut."
            )
            stopped_epoch = epoch
            break

    print("-" * 80)
    history["stopped_epoch"] = stopped_epoch

    if best_model_state is not None:
        checkpoint_payload = {
            "model_state_dict": best_model_state,
            "epoch": best_epoch,
            "val_loss": best_val_loss,
            "train_loss": best_train_loss,
            "lr": best_lr,
            "exp_name": exp_name,
        }

        torch.save(checkpoint_payload, local_checkpoint_file)
        local_size_mb = local_checkpoint_file.stat().st_size / (1024 * 1024)
        print(f"1. Checkpoint lokal tersimpan di {local_checkpoint_file} ({local_size_mb:.2f} MB)")

        if drive_checkpoint_file.exists():
            try:
                drive_checkpoint_file.unlink()
                print("2. File lama di Drive berhasil dibersihkan.")
            except Exception as e:
                print(f"2. Catatan pembersihan Drive: {e}")
        else:
            print("2. Tidak ada file lama yang perlu dibersihkan di Drive.")

        shutil.copy2(local_checkpoint_file, drive_checkpoint_file)
        print(f"3. Checkpoint berhasil disalin ke Google Drive: {drive_checkpoint_file}")

        try:
            verified_payload = torch.load(drive_checkpoint_file, map_location="cpu")
            assert "model_state_dict" in verified_payload
            print(f"4. ✅ VERIFIKASI BERHASIL: Checkpoint Drive valid (Epoch {best_epoch}, Val Loss: {best_val_loss:.4f})")
        except Exception as e:
            print(f"4. ❌ VERIFIKASI GAGAL: Terjadi galat membaca file di Drive: {e}")
            print(f"   Cadangan lokal tersedia di: {local_checkpoint_file}")
    else:
        print("Peringatan: Tidak ada checkpoint yang disimpan.")

    print(f"Log pelatihan tersimpan di : {csv_file}")
    return history
