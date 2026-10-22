"""
Download antiviral drug-virus interaction data from ChEMBL
and train DrugEncoder + CureScoringHead.

Usage:
    python train_drug_encoder.py --download    # Download data first
    python train_drug_encoder.py --train       # Train on downloaded data
"""
import argparse
import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem

from protein_config import DEVICE, SEED, BATCH_SIZE, LR, EPOCHS, GRAD_CLIP
from protein_encoder import ProteinEncoder


DATA_DIR = "data/chembl"
OUTPUT_FILE = os.path.join(DATA_DIR, "antiviral_activities.csv")


def download_chembl_data():
    """Download antiviral activity data from ChEMBL."""
    from chembl_webresource_client.new_client import new_client

    os.makedirs(DATA_DIR, exist_ok=True)

    activity = new_client.activity
    molecule = new_client.molecule
    target = new_client.target

    print("Searching for antiviral targets...")

    # Search for antiviral targets
    virus_keywords = [
        "rabies", "SARS-CoV-2", "HIV", "influenza", "Zika",
        "Hepatitis", "Ebola", "Nipah", "Lassa", "dengue",
        "coronavirus", "herpes", "CMV", "RSV",
    ]

    all_activities = []

    for keyword in virus_keywords:
        print(f"\n  Querying: {keyword}")
        try:
            targets = target.search(keyword)
            target_list = list(targets)
            print(f"    Found {len(target_list)} targets")

            for t in target_list[:5]:  # Top 5 targets per keyword
                chembl_id = t.get('target_chembl_id', '')
                name = t.get('pref_name', '')
                organism = t.get('organism', '')

                # Get activities for this target
                acts = activity.filter(
                    target_chembl_id=chembl_id
                ).filter(
                    standard_type__in=["IC50", "EC50", "GI50", "MIC"],
                    standard_relation="=",
                ).only([
                    'molecule_chembl_id', 'canonical_smiles',
                    'standard_value', 'standard_units',
                    'standard_type', 'target_chembl_id',
                ])

                count = 0
                for a in acts:
                    smiles = a.get('canonical_smiles')
                    value = a.get('standard_value')
                    units = a.get('standard_units')
                    act_type = a.get('standard_type')

                    if smiles and value:
                        try:
                            all_activities.append({
                                'smiles': smiles,
                                'activity': float(value),
                                'units': units,
                                'type': act_type,
                                'target_id': chembl_id,
                                'target_name': name,
                                'organism': organism,
                                'virus_keyword': keyword,
                            })
                            count += 1
                        except (ValueError, TypeError):
                            pass

                    if count >= 500:  # Limit per target
                        break

                print(f"    {name} ({chembl_id}): {count} activities")

                if len(all_activities) > 10000:
                    break

            if len(all_activities) > 10000:
                break

        except Exception as e:
            print(f"    Error: {e}")
            continue

    # Save
    df = pd.DataFrame(all_activities)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df)} activities to {OUTPUT_FILE}")
    return df


class DrugVirusDataset(Dataset):
    """Drug-virus interaction dataset from ChEMBL."""

    def __init__(self, csv_path, fp_size=2048):
        self.fp_size = fp_size
        df = pd.read_csv(csv_path)

        # Convert IC50/EC50 to pActivity (higher = more potent)
        df = df.dropna(subset=['smiles', 'activity'])
        df['activity'] = pd.to_numeric(df['activity'], errors='coerce')
        df = df.dropna(subset=['activity'])

        # Convert to pIC50/pEC50 (negative log)
        df = df[df['activity'] > 0]
        df['pActivity'] = -np.log10(df['activity'] * 1e-9)  # nM to M, then -log10

        # Compute Morgan fingerprints
        self.samples = []
        for _, row in df.iterrows():
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fp_size)
            arr = np.zeros(fp_size, dtype=np.float32)
            for bit in fp.GetOnBits():
                arr[bit] = 1.0

            self.samples.append({
                'fingerprint': arr,
                'pActivity': float(row['pActivity']),
                'smiles': row['smiles'],
                'target': row.get('target_name', 'unknown'),
            })

        print(f"Loaded {len(self.samples)} drug-virus pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'fingerprint': torch.tensor(s['fingerprint'], dtype=torch.float32),
            'pActivity': torch.tensor(s['pActivity'], dtype=torch.float32),
        }


class DrugEncoder(nn.Module):
    """Encodes drug molecules via Morgan fingerprints."""
    def __init__(self, fp_size=2048, output_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(fp_size, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
        )

    def forward(self, fingerprints):
        return F.normalize(self.encoder(fingerprints), dim=-1)


class SimpleCureHead(nn.Module):
    """Predicts drug activity from drug embedding alone (no virus context)."""
    def __init__(self, drug_dim=128):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(drug_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(self, drug_emb):
        return self.scorer(drug_emb).squeeze(-1)


def train_drug_encoder():
    """Train DrugEncoder on ChEMBL antiviral data."""
    torch.manual_seed(SEED)

    csv_path = OUTPUT_FILE
    if not os.path.exists(csv_path):
        print(f"No data found at {csv_path}. Run with --download first.")
        return

    print(f"Loading data from {csv_path}")
    ds = DrugVirusDataset(csv_path)
    if len(ds) < 100:
        print("Too few samples. Need more data.")
        return

    n = len(ds)
    train_n = int(n * 0.8)
    val_n = n - train_n
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_n, val_n])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {train_n} | Val: {val_n}")

    # Model
    drug_encoder = DrugEncoder().to(DEVICE)
    cure_head = SimpleCureHead().to(DEVICE)

    params = list(drug_encoder.parameters()) + list(cure_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)

    best_pearson = -1
    history = []

    print(f"\nTraining DrugEncoder for {EPOCHS} epochs...")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        # Train
        drug_encoder.train()
        cure_head.train()
        train_loss = 0
        n_batch = 0

        for batch in train_loader:
            fps = batch['fingerprint'].to(DEVICE)
            target = batch['pActivity'].to(DEVICE)

            drug_emb = drug_encoder(fps)
            pred = cure_head(drug_emb)
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batch += 1

        # Validate
        drug_encoder.eval()
        cure_head.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                fps = batch['fingerprint'].to(DEVICE)
                target = batch['pActivity'].to(DEVICE)

                drug_emb = drug_encoder(fps)
                pred = cure_head(drug_emb)

                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        val_mse = np.mean((all_preds - all_targets) ** 2)
        val_mae = np.mean(np.abs(all_preds - all_targets))
        pearson = np.corrcoef(all_preds, all_targets)[0, 1] if len(all_preds) > 1 else 0.0

        avg_loss = train_loss / n_batch

        print(f"  Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f} | "
              f"MSE: {val_mse:.4f} | MAE: {val_mae:.4f} | Pearson: {pearson:.4f}")

        if pearson > best_pearson:
            best_pearson = pearson
            torch.save({
                'drug_encoder_state_dict': drug_encoder.state_dict(),
                'cure_head_state_dict': cure_head.state_dict(),
                'val_pearson': pearson,
                'val_mse': val_mse,
                'epoch': epoch,
            }, os.path.join("checkpoints_multivirus", "best_drug_encoder.pt"))
            print(f"    → New best: {pearson:.4f}")

        history.append({
            'epoch': epoch,
            'train_loss': avg_loss,
            'val_mse': val_mse,
            'val_mae': val_mae,
            'val_pearson': pearson,
        })

    print(f"\n{'='*60}")
    print(f"Best Pearson r: {best_pearson:.4f}")
    print(f"Saved to: checkpoints_multivirus/best_drug_encoder.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download ChEMBL data")
    parser.add_argument("--train", action="store_true", help="Train DrugEncoder")
    args = parser.parse_args()

    if args.download:
        download_chembl_data()
    elif args.train:
        train_drug_encoder()
    else:
        print("Specify --download or --train")
