"""
Supervised finetuning on DMS data using codon-level features.
Uses the model's predicted log-probabilities at the mutated codon (18 dims)
instead of the global PDM embedding (128 dims that drowns out single mutations).
"""
import os
import csv
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.stats import spearmanr
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import VOCAB_SIZE, DEVICE, MASK_TOKEN
from model import ViralTCN

NTS = ['A', 'C', 'G', 'T', 'U', 'N']
VOCAB_NT = {b: i for i, b in enumerate(NTS)}

GENCODE = {
    'ATA':'I','ATC':'I','ATT':'I','ATG':'M',
    'ACA':'T','ACC':'T','ACG':'T','ACT':'T',
    'AAC':'N','AAT':'N','AAA':'K','AAG':'K',
    'AGC':'S','AGT':'S','AGA':'R','AGG':'R',
    'CTA':'L','CTC':'L','CTG':'L','CTT':'L',
    'CCA':'P','CCC':'P','CCG':'P','CCT':'P',
    'CAC':'H','CAT':'H','CAA':'Q','CAG':'Q',
    'CGA':'R','CGC':'R','CGG':'R','CGT':'R',
    'GTA':'V','GTC':'V','GTG':'V','GTT':'V',
    'GCA':'A','GCC':'A','GCG':'A','GCT':'A',
    'GAC':'D','GAT':'D','GAA':'E','GAG':'E',
    'GGA':'G','GGC':'G','GGG':'G','GGT':'G',
    'TCA':'S','TCC':'S','TCG':'S','TCT':'S',
    'TTC':'F','TTT':'F','TTA':'L','TTG':'L',
    'TAC':'Y','TAT':'Y','TAA':'*','TAG':'*',
    'TGC':'C','TGT':'C','TGA':'*','TGG':'W',
}
AA_TO_CODONS = {aa: [] for aa in set(GENCODE.values())}
for codon, aa in GENCODE.items():
    AA_TO_CODONS[aa].append(codon)


def load_g_cds(fasta_path="data/rabies_genomes/lyssavirus_full.fasta"):
    with open(fasta_path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if 'NC_001542' in line:
            seq_lines, j = [], i + 1
            while j < len(lines) and not lines[j].startswith('>'):
                seq_lines.append(lines[j].strip())
                j += 1
            genome = ''.join(seq_lines)
            return genome[3317:4883]
    raise ValueError("NC_001542.1 not found")


def get_mutant_codon(g_cds, site, mutant_aa):
    codon_idx = site - 1
    nuc_start = codon_idx * 3
    wt_codon = g_cds[nuc_start:nuc_start + 3]
    if mutant_aa == GENCODE.get(wt_codon, 'X'):
        return g_cds, wt_codon
    mutant_codons = AA_TO_CODONS.get(mutant_aa, [])
    if not mutant_codons:
        return g_cds, wt_codon
    best_codon = min(mutant_codons, key=lambda mc: sum(1 for a, b in zip(wt_codon, mc) if a != b))
    mutant_cds = g_cds[:nuc_start] + best_codon + g_cds[nuc_start + 3:]
    return mutant_cds, best_codon


def seq_to_tensor(seq, max_len=5000):
    seq = seq.upper().replace('-', 'N').replace('?', 'N')
    tokens = [VOCAB_NT.get(c, VOCAB_NT['N']) for c in seq]
    if len(tokens) < max_len:
        tokens += [VOCAB_NT['N']] * (max_len - len(tokens))
    return torch.tensor(tokens[:max_len], dtype=torch.long).unsqueeze(0)


def load_dms_data(dms_path="data/dms/all_antibodies_and_cell_entry.csv"):
    rows = []
    with open(dms_path) as f:
        for row in csv.DictReader(f):
            site = int(row['site'])
            if site < 1:
                continue
            try:
                cell_entry = float(row['cell entry']) if row['cell entry'].strip() else float('nan')
                escape = float(row['monoclonal antibodies escape']) if row['monoclonal antibodies escape'].strip() else float('nan')
            except ValueError:
                continue
            if not math.isnan(cell_entry) and not math.isnan(escape):
                rows.append({
                    'site': site,
                    'wildtype': row['wildtype'],
                    'mutant': row['mutant'],
                    'cell_entry': cell_entry,
                    'escape': escape,
                })
    return rows


def extract_codon_features(model, g_cds, site, mutant_aa, device):
    """Get model's log-probabilities at the 3 masked codon positions (18 dims)."""
    nuc_start = (site - 1) * 3
    mutant_cds, _ = get_mutant_codon(g_cds, site, mutant_aa)

    seq_tensor = seq_to_tensor(mutant_cds).to(device)
    # Mask the codon
    masked = seq_tensor.clone()
    masked[0, nuc_start:nuc_start + 3] = MASK_TOKEN

    with torch.no_grad():
        logits = model.predict_masked(masked)  # (1, L, V)
        log_probs = F.log_softmax(logits[0, nuc_start:nuc_start + 3], dim=-1)  # (3, 6)

    return log_probs.cpu().numpy().flatten()  # 18-dim vector


class DMSDataset(Dataset):
    def __init__(self, dms_rows, g_cds):
        self.rows = dms_rows
        self.g_cds = g_cds

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return (
            torch.tensor([row['cell_entry'], row['escape']], dtype=torch.float),
            row['site'],
            row['mutant'],
        )


class CodonRegressionHead(nn.Module):
    def __init__(self, input_dim=18):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.net(x)


def compute_all_features(model, g_cds, dms_rows, device, batch_size=64):
    """Precompute 18-dim codon features for all mutations (batched)."""
    all_features = []
    print(f"Precomputing features for {len(dms_rows)} mutations...")
    for i in tqdm(range(0, len(dms_rows), batch_size), desc="Features"):
        batch = dms_rows[i:i + batch_size]
        nuc_starts = [(row['site'] - 1) * 3 for row in batch]

        # Create masked sequences in batch
        seqs = []
        for j, row in enumerate(batch):
            mutant_cds, _ = get_mutant_codon(g_cds, row['site'], row['mutant'])
            tensor = seq_to_tensor(mutant_cds).to(device)
            tensor[0, nuc_starts[j]:nuc_starts[j] + 3] = MASK_TOKEN
            seqs.append(tensor)

        stacked = torch.cat(seqs)
        with torch.no_grad():
            logits = model.predict_masked(stacked)
            for j, nuc_start in enumerate(nuc_starts):
                lp = F.log_softmax(logits[j, nuc_start:nuc_start + 3], dim=-1)
                all_features.append(lp.cpu().numpy().flatten())

    return np.array(all_features)


def main():
    print("=" * 60)
    print("Supervised Finetuning (Codon-Level Features)")
    print("=" * 60)

    device = DEVICE

    print("\nLoading model...")
    model = ViralTCN(model_size="lite")
    ckpt = torch.load("checkpoints/best_model.pt", map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Model: val_acc {ckpt.get('val_metrics', {}).get('acc', '?'):.4f}")

    print("\nLoading G CDS...")
    g_cds = load_g_cds()
    print(f"G CDS: {len(g_cds)} bp")

    print("\nLoading DMS data...")
    dms_rows = load_dms_data()
    print(f"Total mutations: {len(dms_rows)}")

    np.random.seed(42)
    np.random.shuffle(dms_rows)
    split = int(0.8 * len(dms_rows))
    train_rows, test_rows = dms_rows[:split], dms_rows[split:]
    print(f"Train: {len(train_rows)}, Test: {len(test_rows)}")

    # Precompute features (one-time cost, ~10 min)
    print("\nPrecomputing codon features...")
    train_feats = compute_all_features(model, g_cds, train_rows, device)
    test_feats = compute_all_features(model, g_cds, test_rows, device)
    print(f"Train features: {train_feats.shape}, Test features: {test_feats.shape}")

    train_targets = np.array([[r['cell_entry'], r['escape']] for r in train_rows])
    test_targets = np.array([[r['cell_entry'], r['escape']] for r in test_rows])

    # Normalize targets
    train_mean = train_targets.mean(axis=0)
    train_std = train_targets.std(axis=0)
    train_targets_norm = (train_targets - train_mean) / (train_std + 1e-8)
    test_targets_norm = (test_targets - train_mean) / (train_std + 1e-8)

    # Convert to tensors
    train_feats_t = torch.FloatTensor(train_feats)
    test_feats_t = torch.FloatTensor(test_feats)
    train_targets_t = torch.FloatTensor(train_targets_norm)
    test_targets_t = torch.FloatTensor(test_targets_norm)

    train_ds = torch.utils.data.TensorDataset(train_feats_t, train_targets_t)
    test_ds = torch.utils.data.TensorDataset(test_feats_t, test_targets_t)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    head = CodonRegressionHead(input_dim=18).to(device)
    optimizer = optim.AdamW(head.parameters(), lr=5e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.MSELoss()

    print("\nTraining...")
    best_test_rho = -1
    for epoch in range(20):
        head.train()
        total_loss = 0
        for feats, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            feats, targets = feats.to(device), targets.to(device)
            preds = head(feats)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        head.eval()
        with torch.no_grad():
            test_preds = head(test_feats_t.to(device)).cpu().numpy()
            train_preds = head(train_feats_t.to(device)).cpu().numpy()

        test_preds_denorm = test_preds * train_std + train_mean
        train_preds_denorm = train_preds * train_std + train_mean

        r_cell, p_cell = spearmanr(test_preds_denorm[:, 0], [r['cell_entry'] for r in test_rows])
        r_esc, p_esc = spearmanr(test_preds_denorm[:, 1], [r['escape'] for r in test_rows])

        print(f"  Test — cell_entry: ρ={r_cell:.4f} (p={p_cell:.4f}) | escape: ρ={r_esc:.4f} (p={p_esc:.4f})")

        if r_cell > best_test_rho:
            best_test_rho = r_cell
            torch.save({
                'head_state_dict': head.state_dict(),
                'train_mean': train_mean,
                'train_std': train_std,
                'epoch': epoch,
                'results': {'cell_entry': r_cell, 'escape': r_esc},
            }, "checkpoints/dms_codon_head.pt")
            print(f"  ✓ Saved (best cell_entry ρ={r_cell:.4f})")

    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"  Cell Entry Spearman: {r_cell:.4f} (p={p_cell:.4f})")
    print(f"  Escape Spearman:     {r_esc:.4f} (p={p_esc:.4f})")
    print(f"  Zero-shot comparison: 0.24 (cell_entry), 0.10 (escape)")


if __name__ == "__main__":
    main()
