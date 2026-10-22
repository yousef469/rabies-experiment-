"""
Train structure predictor on known PDB structures.
Downloads small proteins, computes contact maps from 3D coords, trains contact predictor.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, json, warnings, random
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import Descriptors
warnings.filterwarnings('ignore')

from predict_structure import (
    StructurePredictor, get_esm_embeddings, write_pdb,
    TCNBlock, ContactPredictor, CoordinateReconstructor, DEVICE
)

# ─── PDB protein sequences (small, well-resolved) ───
# Format: (name, sequence, pdb_id)
PROTEIN_DATA = [
    ("BBL Domain", "MRIILLFYLLSVITAIIAIEIPVTEVKCVFSN", "1T8J"),
    ("Villin Headpiece", "DSKGVESTKGATLFKITDDDMVDDFKEGFK", "1VII"),
    ("Alpha3-D", "MRIILTLGVVLLLIVEIPELQASLQDQLK", "1BDD"),
    ("BBA5", "MKVLWAALLVFLGAAGTKAVAVV", "1TUB"),
    ("GTT Domain", "GSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGSEFELRREFHKLPEEVLYEL", "1EMA"),
    ("Acylphosphatase", "MSMTNQPTMCNQKITHEVIAIARELGMN", "1APS"),
    ("CI2", "MEAKAIERLKFNDEMLAAFGNLKTEDE", "2CI2"),
    ("Fyn SH3", "MATGFAELRLPILGHKVESMVKGEDE", "1SHF"),
    ("Crambin", "TTCCPSIVARSNFNVCRLPGTPEAL", "1CRN"),
    ("Rubredoxin", "MFCEDCPYDTPEKLDN", "1IRO"),
    ("WW Domain", "GSKLPPGWEKRSDNGVGYFPRNYLHW", "1I6C"),
    ("Src SH3", "MATGGRYQETVNKLTGEFGKVK", "1SRR"),
    ("Hedgehog", "MVDSGRGQGQHGRGQ", "3M1N"),
    ("HP35", "MRGSHHHHHHGSMELDRPSPTDDL", "1YRF"),
    ("Betabelly", "MKFLIALLHFISAI", "2B3P"),
    ("TrpZip", "GEWTWDDTQSTNN", "1LE1"),
    ("Lambda Repressor", "MVKDTAIAKEVLRQGL", "1LMB"),
    ("Rop", "MAIKWKAEALEIKAAKE", "1ROP"),
    ("Ubiquitin", "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG", "1UBQ"),
    ("BPTI", "RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRTCGGA", "4PTI"),
    ("Lysozyme", "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL", "1LYZ"),
    ("Myoglobin", "MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDKFKHLKSEDEMKASEDLKKHGATVLTALGGILKKKGQHEAEVKPLAESHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGYQG", "1MBN"),
    ("T4 Lysozyme", "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL", "2LZM"),
    ("Nucleocapsid", "MNPNNPSSPQGSDMSIITSPVGDSPGAVN", "1A8C"),
    ("SBD Domain", "MKTLLILAVLCLAQASAQAA", "1A8E"),
    ("Protein G", "MKTIIALSYIFCLVFA", "2GB1"),
]


def download_pdb(pdb_id, cache_dir="pdb_cache"):
    """Download PDB file from RCSB."""
    os.makedirs(cache_dir, exist_ok=True)
    pdb_path = os.path.join(cache_dir, f"{pdb_id}.pdb")

    if os.path.exists(pdb_path):
        return pdb_path

    try:
        from Bio.PDB import PDBList
        pdbl = PDBList()
        pdbl.retrieve_pdb_file(pdb_id, file_format="pdb", pdir=cache_dir, overwrite=True)
        # BioPython saves as pdbXXXX.ent
        src = os.path.join(cache_dir, f"pdb{pdb_id.lower()}.ent")
        if os.path.exists(src):
            os.rename(src, pdb_path)
        elif not os.path.exists(pdb_path):
            return None
        return pdb_path
    except Exception as e:
        print(f"  Failed to download {pdb_id}: {e}")
        return None


def pdb_to_coords(pdb_path):
    """Extract CA coordinates from PDB file."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in [
                    "ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY",
                    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
                    "THR", "TRP", "TYR", "VAL",
                ]:
                    if "CA" in residue:
                        ca = residue["CA"].get_vector().get_array()
                        coords.append(ca)

    return np.array(coords) if coords else None


def compute_contact_map(coords, threshold=8.0):
    """Compute contact map from CA coordinates."""
    n = len(coords)
    dists = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=-1)
    contacts = (dists < threshold).astype(np.float32)
    return contacts


def seq_to_resname(seq):
    """Convert single-letter to three-letter codes."""
    mapping = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
        "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
        "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
        "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    }
    return [mapping.get(aa, "UNK") for aa in seq]


class StructureDataset(Dataset):
    """Dataset of (sequence, contact_map) pairs from PDB structures."""

    def __init__(self, proteins, max_len=200):
        self.data = []
        self.esm_cache = {}

        for name, seq, pdb_id in proteins:
            if len(seq) > max_len:
                continue

            pdb_path = download_pdb(pdb_id)
            if pdb_path is None:
                continue

            coords = pdb_to_coords(pdb_path)
            if coords is None or len(coords) != len(seq):
                continue

            contacts = compute_contact_map(coords)

            self.data.append({
                "name": name,
                "seq": seq,
                "pdb_id": pdb_id,
                "contacts": torch.tensor(contacts, dtype=torch.float32),
                "coords": torch.tensor(coords, dtype=torch.float32),
                "length": len(seq),
            })

        print(f"Loaded {len(self.data)} protein structures from PDB")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_structure_predictor(epochs=30, lr=1e-3, batch_size=4):
    """Train the structure predictor."""
    print("Building dataset from PDB structures...")
    dataset = StructureDataset(PROTEIN_DATA)

    if len(dataset) == 0:
        print("No structures loaded! Check network connection.")
        return None

    model = StructurePredictor(esm_dim=320, hidden=128, contact_blocks=3, coord_blocks=3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params:,}")

    best_acc = 0
    save_path = "checkpoints_multivirus/structure_predictor.pt"

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_pairs = 0

        for sample in dataset:
            seq = sample["seq"]
            contacts_gt = sample["contacts"].unsqueeze(0).to(DEVICE)
            L = len(seq)

            # Get ESM-2 embeddings
            esm_embeds = get_esm_embeddings(seq).to(DEVICE)

            # Forward
            contacts_pred, _ = model(esm_embeds)

            # Pad to max_len if needed
            max_L = max(L, contacts_gt.shape[1])
            if L < max_L:
                pad = torch.zeros(1, max_L, max_L, device=DEVICE)
                pad[:, :L, :L] = contacts_pred
                contacts_pred = pad

            # Binary cross-entropy loss
            loss = F.binary_cross_entropy(contacts_pred[:, :L, :L], contacts_gt[:, :L, :L])

            # Contact accuracy
            with torch.no_grad():
                pred_binary = (contacts_pred[:, :L, :L] > 0.5).float()
                correct = (pred_binary == contacts_gt[:, :L, :L]).sum().item()
                pairs = L * L
                total_correct += correct
                total_pairs += pairs

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataset)
        acc = total_correct / total_pairs if total_pairs > 0 else 0

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "contact_acc": acc,
                "loss": avg_loss,
                "epoch": epoch,
            }, save_path)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Contact Acc: {acc:.3f} | Best: {best_acc:.3f}")

    print(f"\nBest contact accuracy: {best_acc:.3f}")
    print(f"Saved to {save_path}")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    model = train_structure_predictor(epochs=args.epochs, lr=args.lr)
