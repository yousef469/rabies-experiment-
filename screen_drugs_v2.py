"""
Multi-virus drug screening pipeline.
Uses trained protein encoder + mutation encoder + CureScoringHead
to predict cell_entry scores across all viruses with DMS data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import re
import json
import random
from torch.utils.data import Dataset, DataLoader

from protein_config import (
    AA_VOCAB, AA_PAD_TOKEN, AA_MAX_SEQ_LEN, DEVICE,
    BATCH_SIZE, LR, EPOCHS, GRAD_CLIP, SEED, MAX_SAMPLES,
)
from protein_encoder import ProteinEncoder, MutationEncoder


def seq_to_indices(seq, max_len=AA_MAX_SEQ_LEN):
    indices = []
    for aa in seq[:max_len]:
        indices.append(AA_VOCAB.get(aa, 0))
    while len(indices) < max_len:
        indices.append(AA_PAD_TOKEN)
    return torch.tensor(indices, dtype=torch.long)


def mutation_to_tensor(pos, wt, mt):
    wt_idx = AA_VOCAB.get(wt, 0)
    mt_idx = AA_VOCAB.get(mt, 0)
    return torch.tensor([pos, wt_idx, mt_idx], dtype=torch.long)


def parse_mutation(mutation_str):
    match = re.match(r'([A-Z])(\d+)([A-Z*])', mutation_str)
    if not match:
        return None, None, None
    wt, pos, mt = match.groups()
    return int(pos), wt, mt


class MultiVirusCellEntryDataset(Dataset):
    """All viruses with cell_entry DMS data from ViroGym."""

    CELL_ENTRY_FILES = [
        "RABV_cell_entry.csv",
        "SARS_cell_entry_KP311.csv",
        "SARS_cell_entry_Omicron_BA2.csv",
        "SARS_cell_entry_Omicron_XBB15.csv",
        "SARS_cell_entry_Omicron_XBB15_RBD.csv",
        "HIV_cell_entry_B520.csv",
        "HIV_cell_entry_TRO11.csv",
        "LASV_cell_entry.csv",
        "NIPAH_cell_entry.csv",
    ]

    def __init__(self, data_dir, max_len=AA_MAX_SEQ_LEN, max_samples=None):
        self.max_len = max_len
        self.samples = []

        for fname in self.CELL_ENTRY_FILES:
            fpath = os.path.join(data_dir, fname)
            if not os.path.exists(fpath):
                print(f"  Skipping {fname} (not found)")
                continue

            df = pd.read_csv(fpath)
            df = df.dropna(subset=['DMS_score'])
            df['DMS_score'] = pd.to_numeric(df['DMS_score'], errors='coerce')
            df = df.dropna(subset=['DMS_score'])

            virus = fname.split("_")[0]
            count = 0
            for _, row in df.iterrows():
                mutation_str = str(row.get('mutation', ''))
                mutated_seq = str(row.get('mutated_sequence', ''))
                dms_score = float(row['DMS_score'])

                if not mutation_str or not mutated_seq:
                    continue

                pos, wt, mt = parse_mutation(mutation_str)
                if pos is None:
                    continue

                self.samples.append({
                    'sequence': mutated_seq,
                    'position': pos - 1,
                    'wildtype': wt,
                    'mutant': mt,
                    'dms_score': dms_score,
                    'virus': virus,
                })
                count += 1

            print(f"  Loaded {fname}: {count} mutations")

        random.seed(SEED)
        if max_samples and len(self.samples) > max_samples:
            self.samples = random.sample(self.samples, max_samples)

        print(f"  Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'sequence': seq_to_indices(s['sequence'], self.max_len),
            'mutation': mutation_to_tensor(s['position'], s['wildtype'], s['mutant']),
            'dms_score': torch.tensor(s['dms_score'], dtype=torch.float32),
        }


class CureScoringHead(nn.Module):
    def __init__(self, protein_dim=64, mutation_dim=128):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(protein_dim + mutation_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, protein_emb, mutation_emb):
        combined = torch.cat([protein_emb, mutation_emb], dim=-1)
        return self.scorer(combined).squeeze(-1)


class DrugScreeningModel(nn.Module):
    def __init__(self, backbone_checkpoint_path):
        super().__init__()
        self.backbone = ProteinEncoder()
        checkpoint = torch.load(backbone_checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        # Strip "protein_encoder." prefix from MultiVirusModel checkpoint
        cleaned = {k.replace('protein_encoder.', ''): v for k, v in state_dict.items() if k.startswith('protein_encoder.')}
        self.backbone.load_state_dict(cleaned)
        for param in self.backbone.parameters():
            param.requires_grad = False

        protein_dim = self.backbone.pdm.num_subspaces * self.backbone.pdm.subspace_dim
        self.mutation_encoder = MutationEncoder()
        self.cure_head = CureScoringHead(protein_dim=protein_dim, mutation_dim=128)

    def forward(self, seq, mut):
        with torch.no_grad():
            protein_emb = self.backbone.extract_embedding(seq)
        mutation_emb = self.mutation_encoder(mut)
        return self.cure_head(protein_emb, mutation_emb)

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def count_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    val_loss = 0
    val_n = 0

    for batch in loader:
        seq = batch['sequence'].to(DEVICE)
        mut = batch['mutation'].to(DEVICE)
        target = batch['dms_score'].to(DEVICE)

        pred = model(seq, mut)
        val_loss += F.mse_loss(pred, target).item() * target.size(0)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(target.cpu().numpy())
        val_n += target.size(0)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    val_mse = val_loss / val_n
    val_mae = np.mean(np.abs(all_preds - all_targets))
    pearson = np.corrcoef(all_preds, all_targets)[0, 1] if len(all_preds) > 1 else 0.0
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return {'mse': val_mse, 'mae': val_mae, 'pearson': pearson, 'r2': r2}


def train():
    torch.manual_seed(SEED)

    data_dir = "data/viroGym/data/DMS/cleaned_benchmark"
    checkpoint_path = "checkpoints_multivirus/best_pearson.pt"

    print("Loading multi-virus cell_entry data...")
    full_ds = MultiVirusCellEntryDataset(data_dir, max_samples=MAX_SAMPLES)

    n = len(full_ds)
    train_n = int(n * 0.8)
    val_n = n - train_n
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_n, val_n])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {train_n} | Val: {val_n}")

    model = DrugScreeningModel(checkpoint_path).to(DEVICE)
    trainable, total = model.count_parameters()
    print(f"Parameters: {total:,} total ({trainable:,} trainable)")

    best_pearson = -1
    history = []

    # ── Phase 1: Train CureScoringHead (backbone frozen) ────────────────
    print(f"\n{'='*60}")
    print("Phase 1: Training CureScoringHead (backbone frozen)")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=0.01,
    )

    for epoch in range(1, 6):
        model.train()
        train_loss = 0
        n_batch = 0
        for batch in train_loader:
            seq = batch['sequence'].to(DEVICE)
            mut = batch['mutation'].to(DEVICE)
            target = batch['dms_score'].to(DEVICE)

            pred = model(seq, mut)
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batch += 1

        metrics = evaluate(model, val_loader)
        avg_loss = train_loss / n_batch

        print(f"  Epoch {epoch}/5 | Loss: {avg_loss:.4f} | "
              f"MSE: {metrics['mse']:.4f} | Pearson: {metrics['pearson']:.4f} | R²: {metrics['r2']:.4f}")

        if metrics['pearson'] > best_pearson:
            best_pearson = metrics['pearson']
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_metrics': metrics,
                'epoch': epoch,
                'phase': 'frozen',
            }, os.path.join("checkpoints_multivirus", "best_cure_head.pt"))
            print(f"    → New best: {best_pearson:.4f}")

        history.append({'epoch': epoch, 'phase': 'frozen', 'train_loss': avg_loss, **metrics})

    # ── Phase 2: Fine-tune everything ────────────────────────────────────
    print(f"\n{'='*60}")
    print("Phase 2: Fine-tuning backbone + head")
    print(f"{'='*60}")

    model.unfreeze_backbone()
    trainable, total = model.count_parameters()
    print(f"Parameters: {total:,} total ({trainable:,} trainable)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=0.01,
    )

    for epoch in range(6, 11):
        model.train()
        train_loss = 0
        n_batch = 0
        for batch in train_loader:
            seq = batch['sequence'].to(DEVICE)
            mut = batch['mutation'].to(DEVICE)
            target = batch['dms_score'].to(DEVICE)

            pred = model(seq, mut)
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            train_loss += loss.item()
            n_batch += 1

        metrics = evaluate(model, val_loader)
        avg_loss = train_loss / n_batch

        print(f"  Epoch {epoch}/10 (FT) | Loss: {avg_loss:.4f} | "
              f"MSE: {metrics['mse']:.4f} | Pearson: {metrics['pearson']:.4f} | R²: {metrics['r2']:.4f}")

        if metrics['pearson'] > best_pearson:
            best_pearson = metrics['pearson']
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_metrics': metrics,
                'epoch': epoch,
                'phase': 'finetune',
            }, os.path.join("checkpoints_multivirus", "best_cure_head.pt"))
            print(f"    → New best: {best_pearson:.4f}")

        history.append({'epoch': epoch, 'phase': 'finetune', 'train_loss': avg_loss, **metrics})

    # Convert numpy floats for JSON
    for h in history:
        for k, v in h.items():
            if isinstance(v, (np.floating, float)):
                h[k] = float(v)

    with open(os.path.join("checkpoints_multivirus", "cure_head_history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done! Best Pearson r: {best_pearson:.4f}")
    print(f"Saved to: checkpoints_multivirus/best_cure_head.pt")


if __name__ == "__main__":
    train()
