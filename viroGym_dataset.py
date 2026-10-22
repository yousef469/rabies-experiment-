"""
ViroGym DMS dataset loader.
Loads all 79 assays (552K mutations across 13 viruses) for training.
"""
import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from protein_config import (
    AA_VOCAB, AA_PAD_TOKEN, AA_MAX_SEQ_LEN,
    VIRUS_TYPES, PHENOTYPE_CATEGORIES, BATCH_SIZE, NUM_WORKERS, MAX_SAMPLES,
)


def parse_mutation(mutation_str):
    """Parse 'F1A' → (position=1, wildtype='F', mutant='A').
    Handles multi-digit positions and special chars.
    """
    match = re.match(r'([A-Z])(\d+)([A-Z*])', mutation_str)
    if not match:
        return None, None, None
    wt, pos, mt = match.groups()
    return int(pos), wt, mt


def seq_to_indices(seq, max_len=AA_MAX_SEQ_LEN):
    """Convert amino acid string to tensor of indices. Truncates/pads to max_len."""
    indices = []
    for aa in seq[:max_len]:
        indices.append(AA_VOCAB.get(aa, 0))
    # Pad to max_len
    while len(indices) < max_len:
        indices.append(AA_PAD_TOKEN)
    return torch.tensor(indices, dtype=torch.long)


def mutation_to_tensor(pos, wt, mt):
    """Convert mutation to tensor [position, wildtype_idx, mutant_idx]."""
    wt_idx = AA_VOCAB.get(wt, 0)
    mt_idx = AA_VOCAB.get(mt, 0)
    return torch.tensor([pos, wt_idx, mt_idx], dtype=torch.long)


class ViroGymDataset(Dataset):
    """Multi-virus DMS dataset from ViroGym cleaned benchmarks."""

    def __init__(self, data_dir, split="train", max_len=AA_MAX_SEQ_LEN,
                 holdout_virus=None, holdout_phenotype=None):
        """
        Args:
            data_dir: path to cleaned_benchmark/ directory
            split: 'train' or 'val'
            max_len: max sequence length (truncate/pad)
            holdout_virus: if set, exclude this virus for validation
            holdout_phenotype: if set, exclude this phenotype for validation
        """
        self.max_len = max_len
        self.samples = []

        csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

        for csv_file in csv_files:
            filepath = os.path.join(data_dir, csv_file)

            # Determine virus type from filename
            virus = self._get_virus_from_filename(csv_file)
            if virus is None:
                continue

            # Determine phenotype from filename
            phenotype = self._get_phenotype_from_filename(csv_file)
            if phenotype is None:
                continue

            # Holdout logic
            if holdout_virus and virus == holdout_virus:
                continue
            if holdout_phenotype and phenotype == holdout_phenotype:
                continue

            # Load CSV
            try:
                df = pd.read_csv(filepath, nrows=5000 if split == "val" else None)
            except Exception:
                continue

            for _, row in df.iterrows():
                mutation_str = str(row.get('mutation', ''))
                mutated_seq = str(row.get('mutated_sequence', ''))
                dms_score = row.get('DMS_score', None)

                if not mutation_str or not mutated_seq or pd.isna(dms_score):
                    continue

                pos, wt, mt = parse_mutation(mutation_str)
                if pos is None:
                    continue

                self.samples.append({
                    'mutation_str': mutation_str,
                    'position': pos - 1,  # 0-indexed
                    'wildtype': wt,
                    'mutant': mt,
                    'sequence': mutated_seq,
                    'dms_score': float(dms_score),
                    'virus': virus,
                    'virus_idx': VIRUS_TYPES.get(virus, 0),
                    'phenotype': phenotype,
                    'phenotype_idx': PHENOTYPE_CATEGORIES.get(phenotype, 0),
                })

        # Subsample + split train/val (80/20)
        import random
        random.seed(42)
        if len(self.samples) > MAX_SAMPLES:
            self.samples = random.sample(self.samples, MAX_SAMPLES)
        n = len(self.samples)
        split_idx = int(n * 0.8)
        if split == "train":
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

    def _get_virus_from_filename(self, filename):
        """Extract virus type from filename prefix."""
        for virus in VIRUS_TYPES:
            if filename.startswith(virus):
                return virus
        # Check common patterns
        if "SARS" in filename or "sars" in filename:
            return "SCV2"
        if "FLU" in filename or "flu" in filename or "INFA" in filename:
            return "IAV"
        if "HIV" in filename or "hiv" in filename:
            return "HIV"
        if "RABV" in filename or "rabv" in filename:
            return "RABV"
        if "ZIKV" in filename or "zikv" in filename:
            return "ZIKV"
        if "DEN" in filename:
            return "DEN"
        if "NIPAH" in filename:
            return "NIPAH"
        if "LASV" in filename:
            return "LASV"
        if "HBV" in filename:
            return "HBV"
        if "HCV" in filename:
            return "HCV"
        if "AAV2" in filename:
            return "AAV2"
        if "CXB3" in filename:
            return "CXB3"
        if "PESV" in filename:
            return "PESV"
        # Protein-specific names from benchmark.csv
        if "TAT_HV1BR" in filename:
            return "HIV"
        if "I6TAH8" in filename or "NCAP" in filename or "NRAM" in filename or "PA_I34A1" in filename:
            return "IAV"
        return None

    def _get_phenotype_from_filename(self, filename):
        """Extract phenotype category from filename."""
        for pheno in PHENOTYPE_CATEGORIES:
            if pheno in filename.lower():
                return pheno
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'sequence': seq_to_indices(s['sequence'], self.max_len),
            'mutation': mutation_to_tensor(s['position'], s['wildtype'], s['mutant']),
            'dms_score': torch.tensor(s['dms_score'], dtype=torch.float32),
            'virus_idx': torch.tensor(s['virus_idx'], dtype=torch.long),
            'phenotype_idx': torch.tensor(s['phenotype_idx'], dtype=torch.long),
        }


def collate_fn(batch):
    """Stack batch into tensors."""
    return {
        'sequence': torch.stack([b['sequence'] for b in batch]),
        'mutation': torch.stack([b['mutation'] for b in batch]),
        'dms_score': torch.stack([b['dms_score'] for b in batch]),
        'virus_idx': torch.stack([b['virus_idx'] for b in batch]),
        'phenotype_idx': torch.stack([b['phenotype_idx'] for b in batch]),
    }


def get_dataloaders(data_dir, batch_size=BATCH_SIZE, max_len=AA_MAX_SEQ_LEN,
                    holdout_virus=None):
    """Create train and validation dataloaders."""
    train_ds = ViroGymDataset(data_dir, split="train", max_len=max_len,
                               holdout_virus=holdout_virus)
    val_ds = ViroGymDataset(data_dir, split="val", max_len=max_len,
                             holdout_virus=holdout_virus)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    data_dir = "data/viroGym/data/DMS/cleaned_benchmark"
    train_loader, val_loader = get_dataloaders(data_dir, batch_size=4)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Sample batch
    batch = next(iter(train_loader))
    print(f"Sequence shape: {batch['sequence'].shape}")
    print(f"Mutation shape: {batch['mutation'].shape}")
    print(f"DMS scores: {batch['dms_score'][:5]}")
    print(f"Virus indices: {batch['virus_idx'][:5]}")
    print(f"Phenotype indices: {batch['phenotype_idx'][:5]}")
