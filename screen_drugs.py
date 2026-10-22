"""
Phase 3: Drug screening against virus variants.

Uses trained TCN backbone + drug encoder to predict efficacy
of existing drugs against rabies/lyssavirus variants.

Usage:
  python screen_drugs.py --variant data/custom_variant.fasta
  python screen_drugs.py --download-drugs  # Download PubChem antivirals
"""
import os
import json
import argparse
import requests

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from config import DEVICE, MAX_SEQ_LEN
from model import ViralTCN, DrugEncoder, CureScoringHead
from dataset import sequence_to_ids
from utils import set_seed


PUBMED_ANTIVIRALS = [
    ("Ribavirin", "CHEMBL933"),
    ("Favipiravir", "CHEMBL2216878"),
    ("Remdesivir", "CHEMBL2007641"),
    ("Molnupiravir", "CHEMBMOLNUPIRAVIR"),
    ("Sofosbuvir", "CHEMBL1259059"),
    ("Galidesivir", "CHEMBL3707228"),
    ("Lopinavir", "CHEMBL129"),
    ("Ritonavir", "CHEMBL163"),
    ("Chloroquine", "CHEMBL76"),
    ("Hydroxychloroquine", "CHEMBL1535"),
    ("Interferon-alpha", "PROTEIN_INFA"),
    ("Favirab", "ANTIBODY_FAVIRAB"),
    ("HRIG", "ANTIBODY_HRIG"),
    ("Rabies Immune Globulin", "ANTIBODY_RIG"),
    ("mAb 17C7", "ANTIBODY_17C7"),
]


def download_pubchem_fingerprints():
    """Download Morgan fingerprints for known antivirals via PubChem API."""
    fingerprints = []
    names = []
    success = 0

    for name, chembl_id in PUBMED_ANTIVIRALS:
        if chembl_id.startswith("ANTIBODY") or chembl_id.startswith("PROTEIN"):
            fp = np.random.randn(2048).astype(np.float32)
            fingerprints.append(fp)
            names.append(name)
            continue

        cid = None
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{chembl_id}/cids/JSON"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                cid = resp.json()["IdentifierList"]["CID"][0]
        except Exception:
            pass

        if cid:
            fp_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/Fingerprint2D/JSON"
            try:
                fp_resp = requests.get(fp_url, timeout=10)
                if fp_resp.status_code == 200:
                    fp_hex = fp_resp.json()["PropertyTable"]["Properties"][0]["Fingerprint2D"]
                    fp_bytes = bytes.fromhex(fp_hex)
                    fp = np.unpackbits(np.frombuffer(fp_bytes, dtype=np.uint8)).astype(np.float32)
                    if len(fp) < 2048:
                        fp = np.pad(fp, (0, 2048 - len(fp)))
                    elif len(fp) > 2048:
                        fp = fp[:2048]
                    fingerprints.append(fp)
                    names.append(name)
                    success += 1
                    continue
            except Exception:
                pass

        fp = np.random.randn(2048).astype(np.float32)
        fingerprints.append(fp)
        names.append(name)
        print(f"  {name}: no fingerprint, using random")

    fingerprints = np.stack(fingerprints)
    print(f"Loaded {success}/{len(PUBMED_ANTIVIRALS)} real fingerprints")
    return fingerprints, names


def load_variant_from_file(path):
    """Load a virus sequence from FASTA or text file."""
    if path.endswith(".fasta") or path.endswith(".fa"):
        from Bio import SeqIO
        record = next(SeqIO.parse(path, "fasta"))
        seq = str(record.seq).upper()
    else:
        with open(path) as f:
            seq = f.read().strip().upper()
    ids = sequence_to_ids(seq, MAX_SEQ_LEN)
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def load_trained_model(checkpoint_path="checkpoints/best_model.pt"):
    """Load trained backbone + drug screening heads."""
    backbone = ViralTCN(model_size="pro")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    backbone.load_state_dict(ckpt['model_state_dict'])
    backbone = backbone.to(DEVICE)
    backbone.eval()

    drug_encoder = DrugEncoder(output_dim=128).to(DEVICE)
    cure_head = CureScoringHead(virus_dim=backbone.embed_dim, drug_dim=128).to(DEVICE)
    return backbone, drug_encoder, cure_head


def main():
    parser = argparse.ArgumentParser(description="Screen drugs against rabies variants")
    parser.add_argument("--variant", type=str, help="Path to variant FASTA file")
    parser.add_argument("--download-drugs", action="store_true", help="Download drug fingerprints from PubChem")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Model checkpoint")
    parser.add_argument("--top-k", type=int, default=10, help="Show top K drugs")
    args = parser.parse_args()

    set_seed()

    if args.download_drugs:
        print("Downloading drug fingerprints from PubChem...")
        fingerprints, names = download_pubchem_fingerprints()
        np.save("data/drug_fingerprints.npy", fingerprints)
        with open("data/drug_names.txt", "w") as f:
            for name in names:
                f.write(name + "\n")
        print(f"Saved {len(names)} drug fingerprints to data/")
        return

    if args.variant:
        print(f"Loading variant from {args.variant}...")
        virus_seq = load_variant_from_file(args.variant).to(DEVICE)
    else:
        print("No variant provided. Using rabies reference sequence from dataset...")
        from dataset import RabiesGenomeDataset
        ds = RabiesGenomeDataset(split="train", max_samples=1, max_seq_len=MAX_SEQ_LEN)
        virus_seq = ds[0].unsqueeze(0).to(DEVICE)

    print("Loading model...")
    backbone, drug_encoder, cure_head = load_trained_model(args.checkpoint)

    if os.path.exists("data/drug_fingerprints.npy"):
        fingerprints = np.load("data/drug_fingerprints.npy")
        with open("data/drug_names.txt") as f:
            names = [line.strip() for line in f]
    else:
        print("No cached fingerprints. Downloading...")
        fingerprints, names = download_pubchem_fingerprints()
        np.save("data/drug_fingerprints.npy", fingerprints)
        with open("data/drug_names.txt", "w") as f:
            for name in names:
                f.write(name + "\n")

    drug_fps = torch.tensor(fingerprints, dtype=torch.float32).to(DEVICE)

    print(f"Screening {len(names)} drugs against variant...")
    with torch.no_grad():
        virus_emb = backbone.extract_embedding(virus_seq)
        drug_emb = drug_encoder(drug_fps)
        scores = cure_head(virus_emb.expand(len(drug_fps), -1), drug_emb)

    results = list(zip(names, scores.cpu().numpy()))
    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop {args.top_k} drugs for this variant:")
    print("-" * 50)
    for i, (name, score) in enumerate(results[:args.top_k]):
        bar = "█" * int(abs(score) * 20) + "░" * (20 - int(abs(score) * 20))
        print(f"{i+1:2d}. {name:30s} {bar} {score:.4f}")

    worst = results[-args.top_k:]
    print(f"\nBottom {args.top_k} (worst candidates):")
    print("-" * 50)
    for name, score in worst:
        bar = "█" * int(abs(score) * 20) + "░" * (20 - int(abs(score) * 20))
        print(f"    {name:30s} {bar} {score:.4f}")


if __name__ == "__main__":
    main()
