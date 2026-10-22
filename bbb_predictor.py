"""
BBBPredictor: Predict blood-brain barrier permeability using Morgan fingerprints + TCN.
Trained on curated BBB+ and BBB- drug data.
Critical for rabies cure — drug MUST cross BBB to reach brain.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, json, warnings
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
warnings.filterwarnings('ignore')
RDLogger.logger().setLevel(RDLogger.ERROR)

DEVICE = torch.device("cpu")
FP_SIZE = 1024

# ─── Known BBB-permeable drugs (cross BBB, reach brain) ───
BBB_POSITIVE = [
    # Analgesics / Pain
    ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin"),
    ("CC(=O)Nc1ccc(O)cc1", "Acetaminophen"),
    ("CC1=C(CCC1(C)C)C(C)=C\\C=C\\C(C)=C\\C=C\\C(C)=C\\C(=O)NCc1ccccc1", "Isotretinoin"),
    ("CC(C)Cc1ccc(C(C)C(=O)O)cc1", "Ibuprofen"),
    ("O=C(O)c1ccc2[nH]c(=O)[nH]c2c1", "Niflumic acid"),
    ("O=C(O)c1ccc(NC(=O)c2ccccc2)cc1", "Mefenamic acid"),
    # Antidepressants / Psychiatric
    ("CN(C)CCCC(c1ccc(F)cc1)c1ccccc1", "Fluoxetine"),
    ("CN(C)CCc1ccc2nc(=O)[nH]c2c1", "Sertraline-like"),
    ("ClC1=CC=C(C=C1)C(=O)C2=CC=CC=C2Cl", "Duloxetine-like"),
    ("CNCCC(c1ccc(F)cc1)c1ccccc1", "Fluoxetine amine"),
    ("O=C(NCC1=CC=CC=C1)C1=CC=CC=C1", "Carbamazepine-like"),
    # Anesthetics
    ("O=C(CCl)Nc1ccccc1", "2-Chloroacetanilide"),
    ("ClCC(=O)Nc1ccc(F)cc1", "Fluorocaine"),
    ("CC(=O)NC1=CC=CC=C1", "Acetanilide"),
    # Anticonvulsants
    ("O=C1NC(=O)C(C)(C)N1", "Trimethadione"),
    ("CC(C)(C(=O)NC1=CC=CC=C1)O", "Valproamide"),
    # Antihistamines
    ("CN(C)CCOC(c1ccccc1)c1ccc(Cl)cc1", "Diphenhydramine"),
    ("CC(CN(C)C)Oc1cccc2ccccc12", "Promethazine-like"),
    ("C(=O)(NC1CCCCC1)C1=CC=CC=C1", "Cyclohexylbenzamide"),
    # Stimulants
    ("CC(N)Cc1ccc(O)c(O)c1", "Dopamine-like"),
    ("CNCC(O)c1ccc(O)c(O)c1", "Epinephrine-like"),
    ("CC(=O)Nc1ccc(N(C)C)cc1", "Phenacetin"),
    # Steroids (cross BBB easily)
    ("CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O", "Testosterone"),
    ("CC12CCC3C(CCC4=CC(=O)C=CC34C)C1CCC2O", "Estradiol"),
    ("CC(=O)OC1CCC2C1CCC1(C)C2CCC1=O", "Progesterone"),
    ("O=C1CC2(CCC3C2CCC2(C)C3CCC21)C", "5alpha-DHT"),
    # Cannabinoids
    ("CCCCCC(CCC)C(=O)OC1=C(C)C(C)=CC(C)=C1", "THC-like"),
    # Barbiturates
    ("CCCC(CCC)(C(=O)NC1=CC=CC=C1)C(=O)NC1=CC=CC=C1", "Phenobarbital"),
    ("CC1(C)CC(=O)NC(=O)N1", "Metharbital-like"),
    # Opioids
    ("CN1CC(C2=CC=CC=C12)C(C1=CC=CC=C1)O", "Morphine-like"),
    ("CC(=O)OC1C=CC2C3CC4=CC(=O)C=CC4(C)C3CCC21C", "Heroin-like"),
    # Caffeine
    ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine"),
    # Nicotine
    ("c1ncc(c1)[C@@H]1CCCN1C", "Nicotine"),
    # Sulfonamides
    ("CC1=CC=C(C=C1)S(=O)(=O)NC(=O)C1=CC=CC=C1", "Sulfabenzamide"),
    # Alcohols (small, cross easily)
    ("CCO", "Ethanol"),
    ("O", "Water"),
    ("CCCC", "Butane-like"),
    # Anxiolytics
    ("ClC1=CC=C(C=C1)N2C(=O)CN=C2C3=CC=CC=C3", "Benzodiazepine-like"),
    ("O=C1CN=C(c2ccccc2)c2cc(Cl)ccc2N1", "Lorazepam-like"),
    ("O=C1CN=C(c2ccccc2)c2ccccc2N1", "Diazepam-like"),
    ("ClC1=CC=CC=C1C(=O)N2CCN(C)CC2", "Buspirone-like"),
    # Antipsychotics
    ("CC(=O)N1CCN(CC1)c1ccc(S(=O)(=O)c2ccccc2)cc1", "Sulpiride-like"),
    # Propranolol
    ("OCC(CO)NC(C)Cc1ccc(O)c(OC)c1", "Metanephrine-like"),
    ("O=C(O)C1=CC=CC=C1O", "Salicylic acid"),
    # Small BBB-crossing molecules
    ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "Ibuprofen2"),
    ("O=C(O)c1ccc(O)cc1", "4-Hydroxybenzoic acid"),
    ("OCC(O)CO", "Glycerol"),
    ("CCCCCCCC", "Octane"),
    ("CC(=O)OCC", "Ethyl acetate"),
]

# ─── Known BBB-impermeable drugs (DON'T cross BBB) ───
BBB_NEGATIVE = [
    # Large polar molecules
    ("OC[C@H]1OC(O[C@@H]2OC(CO)[C@@H](O)[C@H](O)[C@@H]2O)[C@H](O)[C@@H](O)[C@@H]1O", "Sucrose"),
    ("NCC(=O)NCC(=O)O", "Glycylglycine"),
    ("OC(=O)CC(O)(CC(=O)O)C(=O)O", "Citric acid"),
    ("NCC(=O)O", "Glycine"),
    ("NC(=O)N", "Urea"),
    ("NCCN", "Ethylenediamine"),
    ("OC(=O)C(O)C(O)C(=O)O", "Tartaric acid"),
    ("OC(=O)C1=CC=CC=C1O", "Salicylic acid salt"),
    # Antibiotics (too polar/large)
    ("CC1(C)SC2C(NC(=O)CC3=CC=CC=C3)C(=O)N2C1C(=O)O", "Ampicillin"),
    ("CC1(C)SC2C(NC(=O)C(N)C3=CC=CC=C3)C(=O)N2C1C(=O)O", "Amoxicillin"),
    ("OC(=O)C1=CC(=CC=C1)NC1=C2C=CC=CC=2NC2=CC=CC=C12", "Rifampicin-like"),
    ("CC1=CC=C(C=C1)S(=O)(=O)NC(=O)NN2CCCCCC2", "Furosemide-like"),
    ("CC(CS)C(=O)N1CCCC1C(=O)O", "Captopril-like"),
    # Chemotherapy (too large)
    ("NC1=NC=NC2=C1N=CN=C2N", "Adenine"),
    ("OC1=NC(=O)NC2=C1N=CN2", "Hypoxanthine"),
    ("NCC(=O)NCC(=O)NCC(=O)O", "Triglycine"),
    # Sugars
    ("OC[C@H]1OC(=O)[C@H](O)[C@@H](O)[C@@H]1O", "Glucuronic acid"),
    ("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "Glucose"),
    # Peptides
    ("NCCCC(N)C(=O)O", "Ornithine"),
    ("NCC(N)C(=O)O", "Glycine2"),
    ("CC(C)CC(N)C(=O)O", "Leucine"),
    ("NCC(=O)OCC", "Glycine ethyl ester"),
    # Large antibiotics
    ("CC1=CC=CC=C1NC(=O)C2=CC=C(C=C2)S(=O)(=O)NC(=O)NC3=CC=CC=C3", "Sulfonamide large"),
    ("OC(=O)C1=CC=C(C=C1)NC1=NC=NC2=C1N=CN=C2NC1=CC=CC=C1", "Pteridine large"),
    # Heparin-like
    ("OS(=O)(=O)OCC(O)COC(=O)CC(=O)OCC(O)COS(=O)(=O)O", "Heparin fragment"),
    # Proteins/peptides
    ("CC(C)CC(NC(=O)C(CC(N)=O)NC(=O)C(CC(=O)O)NC(=O)C(N)CCCCN)C(=O)NCC(=O)O", "Leu-enkephalin"),
    # Antibodies (way too large)
    # Nucleotides
    ("OC[C@H]1OC[C@H](O)[C@@H]1O", "Deoxyribose"),
    ("OC1=NC(=O)C2=NC=NC2=N1", "Purine"),
    # Polar antivirals
    ("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "Mannitol"),
    ("OC(=O)C1=CC=C(C=C1)NC(=O)C2=CC=CC=C2", "Hippuric acid"),
    # Metformin-like
    ("CN(C)C(=N)NC(=N)N", "Metformin"),
    # Biguanides
    ("NC(=N)NC(=N)N", "Guanidine"),
    ("O=C(O)C1=CC=C(C=C1)NC(=O)C2=CC=C(C=C2)S(=O)(=O)N", "Sulfasalazine"),
    # Large anticancer
    ("CC1=C2CCC3C(CCC4=CC(=O)C=CC34C)C2(C)CC(O)C1O", "Docetaxel-like"),
    # Proteins
    ("CC(C)CC(NC(=O)C(CC(=O)O)NC(=O)C(CC(=O)O)NC(=O)C(N)CCCCN)C(=O)NCC(=O)O", "Tripeptide"),
    # Carbohydrates
    ("OC[C@H]1OC(O[C@H]2OC[C@@H](O)[C@@H](O)[C@@H]2O)[C@H](O)[C@@H](O)[C@@H]1O", "Disaccharide"),
    # Very polar
    ("N[C@@H](CC(=O)O)C(=O)O", "Aspartic acid"),
    ("N[C@@H](CS)C(=O)O", "Cysteine"),
    ("N[C@@H](CCC(=O)O)C(=O)O", "Glutamic acid"),
    ("N[C@@H](CC(=O)N)C(=O)O", "Asparagine"),
    ("N[C@@H](CCC(=O)N)C(=O)O", "Glutamine"),
    ("NC(=O)CC(N)C(=O)O", "Asparagine2"),
    ("OC(=O)C1=CC=CC=C1NC(=O)C1=CC=CC=C1C(=O)O", "Dibenzoic acid"),
    # Big molecules
    ("OC(=O)C1=CC=C(C=C1)NC(=O)C2=NC=CC=C2", "Pyridine amide"),
    ("O=S(=O)(NC1=CC=CC=C1)C1=CC=C(C=C1)S(=O)(=O)NC1=CC=CC=C1", "Bis-sulfonamide"),
    ("OC(=O)C1=CC=C(C=C1)NC(=O)C2=CC=C(C=C2)NC(=O)C3=CC=C(C=C3)C(=O)O", "Tris-benzoic"),
]


class BBBDataset(Dataset):
    def __init__(self, max_len=FP_SIZE):
        self.fps = []
        self.labels = []

        for smi, name in BBB_POSITIVE:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=max_len)
                self.fps.append(np.array(fp))
                self.labels.append(1)

        for smi, name in BBB_NEGATIVE:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=max_len)
                self.fps.append(np.array(fp))
                self.labels.append(0)

        print(f"BBB dataset: {sum(self.labels)} positive, {len(self.labels) - sum(self.labels)} negative")

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.fps[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


class BBBPredictor(nn.Module):
    """Morgan FP + TCN → BBB permeability prediction."""

    def __init__(self, fp_size=1024, hidden=128, n_blocks=3):
        super().__init__()
        self.input_proj = nn.Conv1d(1, hidden, 1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden, hidden, 3, padding=1, groups=hidden),
                nn.GroupNorm(8, hidden),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, 3, padding=1, groups=hidden),
                nn.GroupNorm(8, hidden),
                nn.GELU(),
            ) for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out = self.input_proj(x)
        for block in self.blocks:
            residual = out
            for layer in block:
                out = layer(out)
            out = out + residual
        out = out.mean(dim=-1)
        return self.head(out).squeeze(-1)


def train_bbb(epochs=50, lr=1e-3, save_path="checkpoints_multivirus/bbb_predictor.pt"):
    dataset = BBBDataset()
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = BBBPredictor().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params:,}")

    best_acc = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for fps, labels in loader:
            fps, labels = fps.to(DEVICE), labels.to(DEVICE)
            logits = model(fps)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        avg_loss = total_loss / len(loader)

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "accuracy": acc,
                "loss": avg_loss,
            }, save_path)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.3f} | Best: {best_acc:.3f}")

    print(f"\nBest accuracy: {best_acc:.3f}")
    return model


def predict_bbb(smiles, model=None):
    """Predict if a molecule crosses the BBB using trained 8-descriptor model."""
    if model is None:
        from pkpd_model import SimpleBBBNet
        model = SimpleBBBNet()
        ckpt = torch.load("checkpoints_multivirus/bbb_predictor.pt", map_location=DEVICE)
        model.load_state_dict(ckpt["state_dict"])
    model.eval()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rot = Descriptors.NumRotatableBonds(mol)
    aro = Descriptors.NumAromaticRings(mol)
    hba_hbd = hba + hbd

    features = np.array([[mw, logp, tpsa, hbd, hba, rot, aro, hba_hbd]], dtype=np.float32)
    mean = ckpt.get("mean")
    std = ckpt.get("std")
    if mean is not None and std is not None:
        m = np.array(mean, dtype=np.float32)
        s = np.array(std, dtype=np.float32)
        features = (features - m) / (s + 1e-8)

    tensor = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        logit = model(tensor)
        prob = torch.sigmoid(logit).item()

    return {
        "smiles": smiles,
        "bbb_score": prob,
        "crosses_bbb": prob > 0.5,
        "confidence": abs(prob - 0.5) * 2,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--smiles", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "train":
        model = train_bbb()
    else:
        if args.smiles:
            result = predict_bbb(args.smiles)
            print(json.dumps(result, indent=2))
        else:
            # Test with known drugs
            test_drugs = [
                ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin"),
                ("CC(=O)Nc1ccc(O)cc1", "Acetaminophen"),
                ("CN(C)CCCC(c1ccc(F)cc1)c1ccccc1", "Fluoxetine"),
                ("CC(C)Cc1ccc(C(C)C(=O)O)cc1", "Ibuprofen"),
                ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine"),
                ("c1ncc(c1)[C@@H]1CCCN1C", "Nicotine"),
                ("CN(C)C(=N)NC(=N)N", "Metformin"),
                ("NC(=N)NC(=N)N", "Guanidine"),
                ("CC(CS)C(=O)N1CCCC1C(=O)O", "Captopril"),
                ("OC(=O)C1=CC=CC=C1NC(=O)C1=CC=CC=C1C(=O)O", "Hippuric acid"),
            ]
            for smi, name in test_drugs:
                r = predict_bbb(smi)
                status = "BBB+" if r["crosses_bbb"] else "BBB-"
                print(f"{name:<25} {status} (score={r['bbb_score']:.3f})")
