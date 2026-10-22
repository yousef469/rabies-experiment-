"""
Utilities for training, metrics, checkpointing, and data helpers.
"""
import os
import random
import math

import torch
import numpy as np

from config import DEVICE, SEED


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_accuracy(preds, labels, ignore_index=-100):
    mask = labels != ignore_index
    if mask.sum() == 0:
        return 0.0
    return (preds[mask] == labels[mask]).float().mean().item()


class Checkpointer:
    def __init__(self, checkpoint_dir="checkpoints", monitor="val_loss", mode="min"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')

    def save(self, model, optimizer, scheduler, epoch, val_metrics, name="best_model"):
        path = os.path.join(self.checkpoint_dir, f"{name}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_metrics': val_metrics,
        }, path)

    def step(self, model, optimizer, scheduler, epoch, val_metrics):
        current = val_metrics.get(self.monitor, 0)
        improved = (self.mode == 'min' and current < self.best_score) or \
                   (self.mode == 'max' and current > self.best_score)
        if improved:
            self.best_score = current
            self.save(model, optimizer, scheduler, epoch, val_metrics, "best_model")
            return True
        return False

    def load_best(self, model, device=DEVICE):
        path = os.path.join(self.checkpoint_dir, "best_model.pt")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
            return ckpt.get('val_metrics', {})
        return {}
