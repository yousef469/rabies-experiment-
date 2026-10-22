"""
Multi-virus training script for VirusFighter platform.
Trains ProteinEncoder + MutationEncoder + DMSHead on ViroGym DMS data.

Usage:
    python train_multivirus.py
    python train_multivirus.py --epochs 50 --batch-size 64 --holdout-virus HIV
"""
import argparse
import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np

from protein_config import (
    DEVICE, EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY,
    WARMUP_STEPS, GRAD_CLIP, SEED, EMBED_DIM,
)
from protein_encoder import MultiVirusModel
from viroGym_dataset import get_dataloaders


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup then cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    total_mse = 0
    total_samples = 0
    batch_count = 0

    for batch in loader:
        seq = batch['sequence'].to(device)
        mut = batch['mutation'].to(device)
        target = batch['dms_score'].to(device)

        optimizer.zero_grad()
        pred = model(seq, mut)

        # MSE loss on DMS scores
        loss = F.mse_loss(pred, target)

        # Add orthogonality regularization from PDM
        ortho_loss = model.protein_encoder.pdm.orthogonality_loss() * 0.01

        total = loss + ortho_loss
        total.backward()

        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()
        scheduler.step()

        total_loss += total.item()
        total_mse += loss.item()
        total_samples += target.size(0)
        batch_count += 1

        if batch_count % 100 == 0:
            avg_loss = total_loss / batch_count
            avg_mse = total_mse / batch_count
            lr = optimizer.param_groups[0]['lr']
            print(f"  [Epoch {epoch}] Batch {batch_count}/{len(loader)} | "
                  f"Loss: {avg_loss:.4f} | MSE: {avg_mse:.4f} | LR: {lr:.6f}")

    return total_loss / max(batch_count, 1), total_mse / max(batch_count, 1)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_mse = 0
    total_mae = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    for batch in loader:
        seq = batch['sequence'].to(device)
        mut = batch['mutation'].to(device)
        target = batch['dms_score'].to(device)

        pred = model(seq, mut)
        total_mse += F.mse_loss(pred, target).item() * target.size(0)
        total_mae += F.l1_loss(pred, target).item() * target.size(0)
        total_samples += target.size(0)

        all_preds.append(pred.cpu().numpy())
        all_targets.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Pearson correlation
    if len(all_preds) > 1:
        corr = np.corrcoef(all_preds, all_targets)[0, 1]
    else:
        corr = 0.0

    # R^2 score
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return {
        'mse': total_mse / max(total_samples, 1),
        'mae': total_mae / max(total_samples, 1),
        'pearson_r': corr,
        'r2': r2,
        'n_samples': total_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Train multi-virus DMS predictor")
    parser.add_argument("--data-dir", default="data/viroGym/data/DMS/cleaned_benchmark",
                        help="Path to ViroGym cleaned benchmarks")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--holdout-virus", default=None,
                        help="Virus to hold out for validation (e.g., 'HIV')")
    parser.add_argument("--checkpoint-dir", default="checkpoints_multivirus",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    set_seed(SEED)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Loading ViroGym data from: {args.data_dir}")

    train_loader, val_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size,
        holdout_virus=args.holdout_virus,
    )

    print(f"Train batches: {len(train_loader)} ({len(train_loader.dataset)} samples)")
    print(f"Val batches: {len(val_loader)} ({len(val_loader.dataset)} samples)")

    # Create model
    model = MultiVirusModel().to(DEVICE)
    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # Resume if specified
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resumed from checkpoint: {args.resume} (epoch {start_epoch})")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_lr_scheduler(optimizer, WARMUP_STEPS, total_steps)

    # Training loop
    best_val_loss = float('inf')
    best_pearson = -1
    history = []

    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 70)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_mse = train_epoch(
            model, train_loader, optimizer, scheduler, DEVICE, epoch
        )

        val_metrics = validate(model, val_loader, DEVICE)

        elapsed = time.time() - t0

        print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.1f}s)")
        print(f"  Train — Loss: {train_loss:.4f} | MSE: {train_mse:.4f}")
        print(f"  Val   — MSE: {val_metrics['mse']:.4f} | MAE: {val_metrics['mae']:.4f} | "
              f"Pearson r: {val_metrics['pearson_r']:.4f} | R²: {val_metrics['r2']:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics,
            'train_loss': train_loss,
            'config': {
                'embed_dim': EMBED_DIM,
                'n_params': n_params,
                'holdout_virus': args.holdout_virus,
            }
        }

        torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'last.pt'))

        if val_metrics['mse'] < best_val_loss:
            best_val_loss = val_metrics['mse']
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best_mse.pt'))
            print(f"  → New best MSE: {best_val_loss:.4f}")

        if val_metrics['pearson_r'] > best_pearson:
            best_pearson = val_metrics['pearson_r']
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best_pearson.pt'))
            print(f"  → New best Pearson r: {best_pearson:.4f}")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_mse': val_metrics['mse'],
            'val_mae': val_metrics['mae'],
            'val_pearson': val_metrics['pearson_r'],
            'val_r2': val_metrics['r2'],
        })

        print()

    # Save history
    with open(os.path.join(args.checkpoint_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print("=" * 70)
    print(f"Training complete!")
    print(f"Best MSE: {best_val_loss:.4f}")
    print(f"Best Pearson r: {best_pearson:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}/")


if __name__ == "__main__":
    main()
