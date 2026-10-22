"""
Self-supervised pretraining + fine-tuning for viral sequence understanding.

Phase 1: Masked language modeling on lyssavirus genomes
Phase 2: Fine-tune on protein region prediction
Phase 3 (optional): Candidate screening with drug molecules
"""
import os
import sys
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from config import (
    VOCAB_SIZE, EPOCHS, LR, WEIGHT_DECAY, WARMUP_STEPS,
    BATCH_SIZE, DEVICE, MODEL_SIZE, MASK_TOKEN, MAX_SEQ_LEN,
)
from model import ViralTCN, RabiesDefenseModel
from dataset import create_dataloaders
from utils import (
    set_seed, count_parameters, get_cosine_schedule_with_warmup,
    compute_accuracy, Checkpointer,
)


class Trainer:
    def __init__(self, model, train_loader, val_loader, device=DEVICE):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.95),
        )

        total_steps = len(train_loader) * EPOCHS
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=WARMUP_STEPS,
            num_training_steps=total_steps,
            min_lr_ratio=0.01,
        )

        self.checkpointer = Checkpointer("checkpoints", monitor="acc", mode="max")
        self.best_val_loss = float('inf')
        self.scaler = torch.cuda.amp.GradScaler() if 'cuda' in device else None

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        start_time = time.time()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for batch_idx, (masked_seq, labels) in enumerate(pbar):
            masked_seq = masked_seq.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.scaler:
                with torch.cuda.amp.autocast():
                    logits = self.model.predict_masked(masked_seq)
                    logits = logits.reshape(-1, VOCAB_SIZE)
                    labels_flat = labels.reshape(-1)
                    loss = self.criterion(logits, labels_flat)
                    orth_loss = self.model.pdm.orthogonality_loss()
                    total_loss_val = loss + 0.05 * orth_loss
                self.scaler.scale(total_loss_val).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model.predict_masked(masked_seq)
                logits = logits.reshape(-1, VOCAB_SIZE)
                labels_flat = labels.reshape(-1)
                loss = self.criterion(logits, labels_flat)
                orth_loss = self.model.pdm.orthogonality_loss()
                total_loss_val = loss + 0.05 * orth_loss
                total_loss_val.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            self.scheduler.step()

            total_loss += total_loss_val.item() * masked_seq.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += compute_accuracy(preds, labels_flat, -100) * (labels_flat != -100).sum().item()
            total_tokens += (labels_flat != -100).sum().item()

            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                "loss": f"{total_loss_val.item():.4f}",
                "lr": f"{current_lr:.2e}",
            })

        avg_loss = total_loss / len(self.train_loader.dataset)
        avg_acc = total_correct / max(total_tokens, 1)
        epoch_time = time.time() - start_time
        return avg_loss, avg_acc, epoch_time

    @torch.no_grad()
    def evaluate(self, loader, name="Val"):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        for masked_seq, labels in tqdm(loader, desc=f"[{name}]"):
            masked_seq = masked_seq.to(self.device)
            labels = labels.to(self.device)

            logits = self.model.predict_masked(masked_seq)
            logits = logits.reshape(-1, VOCAB_SIZE)
            labels_flat = labels.reshape(-1)
            loss = self.criterion(logits, labels_flat)

            total_loss += loss.item() * masked_seq.size(0)
            preds = logits.argmax(dim=-1)
            mask = labels_flat != -100
            total_correct += (preds[mask] == labels_flat[mask]).sum().item()
            total_tokens += mask.sum().item()

        avg_loss = total_loss / len(loader.dataset)
        avg_acc = total_correct / max(total_tokens, 1)
        return {"loss": avg_loss, "acc": avg_acc}

    def resume_from_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            return 0
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if self.scheduler and ckpt.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resumed from epoch {ckpt['epoch']+1}/{EPOCHS} (val_acc={ckpt.get('val_metrics', {}).get('acc', 0):.4f})")
        return start_epoch

    def fit(self, start_epoch=0):
        for epoch in range(start_epoch, EPOCHS):
            train_loss, train_acc, epoch_time = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader, "Val")

            current_lr = self.optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            improved = self.checkpointer.step(
                self.model, self.optimizer, self.scheduler, epoch, val_metrics
            )
            if improved:
                print(f"  ✓ Best model saved (val_acc={val_metrics['acc']:.4f})")

            self.checkpointer.save(
                self.model, self.optimizer, self.scheduler, epoch, val_metrics, "last_model"
            )

        print("\n=== Training Complete ===")
        final_val = self.evaluate(self.val_loader, "Final Val")
        print(f"Final — Loss: {final_val['loss']:.4f} | Acc: {final_val['acc']:.4f}")
        return final_val


def main(model_size=None):
    set_seed()

    if model_size is None:
        model_size = MODEL_SIZE

    print(f"Device: {DEVICE}")
    print(f"Model size: {model_size}")

    model = ViralTCN(model_size=model_size)
    trainable, total = count_parameters(model)
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    print("\nCreating dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(
        fasta_path="data/rabies_genomes/lyssavirus_full.fasta",
        max_samples=None, max_seq_len=5000, batch_size=16 if model_size == "pro" else 32,
    )

    trainer = Trainer(model, train_loader, val_loader)

    # Auto-resume from latest checkpoint
    import os
    ckpt_dir = trainer.checkpointer.checkpoint_dir
    best_path = os.path.join(ckpt_dir, "best_model.pt")
    last_path = os.path.join(ckpt_dir, "last_model.pt")
    resume_path = best_path if os.path.exists(best_path) else (last_path if os.path.exists(last_path) else None)

    start_epoch = 0
    if resume_path:
        start_epoch = trainer.resume_from_checkpoint(resume_path)
        if start_epoch > 0:
            print(f"Resumed from epoch {start_epoch}")

    trainer.fit(start_epoch=start_epoch)


if __name__ == "__main__":
    main()
