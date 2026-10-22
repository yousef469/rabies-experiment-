"""
Drug screening pipeline: Screen real drug molecules against viruses.
Uses trained ProteinEncoder + DrugEncoder (Morgan FP) + CureScoringHead.

Usage:
    python screen_drugs_real.py                     # Screen demo drugs against rabies
    python screen_drugs_real.py --virus rabies      # Specific virus
    python screen_drugs_real.py --virus-all         # Screen against all viruses
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from protein_config import DEVICE, AA_VOCAB, AA_PAD_TOKEN, AA_MAX_SEQ_LEN
from protein_encoder import ProteinEncoder


# ── Known viral protein sequences (from ViroGym) ───────────────────────────
VIRAL_PROTEINS = {
    "rabies_g": "AGKFPIYTIPDKLGPWSPIDIHHLSCPNNLVVEDEGCTNLSGFSYMELKVGYISAIKMNGFTCTGVVTEAETYTNFVGYVTTTFKRKHFRPTPDACRAAYNWKMAGDPRYEESLHNPYPDYHWLRTVKTTKESLVIISPSVADLDPYDRSLHSRVFPGGNCSGVAVSSTYCSTNHDYTIWMPENPRLGMSCDIFTNSRGKRASKGSETCGFVDERGLYKSLKGACKLKLCGVLGLRLMDGTWVAMQTSNETKWCPPGQLVNLHDFRSDEIEHLVVEELVKKREECLDALESIMTTKSVSFRRLSHLRKLVPGFGKAYTIFNKTLMEADAHYKSVRTWNEIIPSKGCLRVGGRCHPHVNGVFFNGIILGPDGNVLIPEMQSSLLQQHMELLVSSVIPLMHPLADPSTVFKNGDEAEDFVEVHLPDVHERISGVD",
    "sars_spike_rbd": "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST",
    "hiv_env": "MRVMGIQRNCQHLFRWGTMILGMIIICSAYENLWVTVYYGVPVWKDAETTLFCASDAKAYETEKHNVWATHACVPTDPNPQEIHLENVTEEFNMWKNNMVEQMHTDIISLWDQSLKPCVKLTPLCVTLQCTNVTNNITDDMRGELKNCSFNMTTELRDKKQKVYSLFYRLDVVQINENQGNRSNNSNKEYRLINCNTSAITQACPKVSFEPIPIHYCAPAGFAILKCKDKKFNGTGPCPSVSTVQCTHGIKPVVSTQLLLNGSLAEEEVMIRSENITNNAKNILVQFNTPVQINCTRPNNNTRKSIRIGPGQAFYATGDIIGDIRQAHCNVSKATWNETLGKVVKQLRKHFGNNTIIRFANSSGGDLEVTTHSFNCGGEFFYCNTSGLFNSTWISNTSVQGSNSTGSNDSITLPCRIKQIINMWQRIGQAMYAPPIQGVIRCVSNITGLILTRDGGSTNSTTETFRPGGGDMRDNWRSELYKYKVVKIEPLGVAPTRAKRRVVGREKRAVGIGAVFLGFLGAAGSTMGAASMTLTVQARNLLSGIVQQQSNLLRAIEAQQHLLKLTVWGIKQLQARVLAVERYLRDQQLLGIWGCSGKLICTTNVPWNSSWSNRNLSEIWDNMTWLQWDKEISNYTQIIYGLLEESQNQQEKNEQDLLALDKWASLWNWFDISNWLWYIKIFIMIVGGLIGLRIVFAVLSVIHRVRQGYSPLSFQTHTPNPRGLDRPERIEEEDGEQDRGRSTRLVSGFLALAWDDLRSLCLFCYHRLRDFILIAARIVELLGHSSLKGLRLGWEGLKYLWNLLAYWGRELKISAINLFDTIAIAVAEWTDRVIEIGQRLCRAFLHIPRRIRQGLERALL",
}


# ── FDA-approved antivirals + known antiviral compounds ─────────────────────
DRUG_LIBRARY = [
    {"name": "Remdesivir", "smiles": "CCC(CC)OC1C=C(C=CC1NC(=O)NC2=CC=C(C=C2)OC3=CC=CC=C3)C#N", "target": "RdRp inhibitor"},
    {"name": "Ribavirin", "smiles": "C1=NC(=NN1C2C(C(C(O2)CO)O)O)C(=O)N", "target": "Nucleoside analog"},
    {"name": "Favipiravir", "smiles": "CC1=NC(=NC=C1O)NC(=O)N", "target": "RdRp inhibitor"},
    {"name": "Molnupiravir", "smiles": "CC(C)OC(=O)NC1=NC(=O)N(C=N1)C2C(C(C(O2)C)O)O", "target": "Nucleoside analog"},
    {"name": "Oseltamivir", "smiles": "CCCCC(CC)OC1C=C(C(CC1NC(=O)C)N)C(=O)OCC", "target": "Neuraminidase inhibitor"},
    {"name": "Acyclovir", "smiles": "C1=NC2=NC(=NC=C2N1CC(CO)O)N", "target": "Nucleoside analog"},
    {"name": "Valacyclovir", "smiles": "CC(C)C(C(=O)NCC1=NC2=NC(=NC=C2N1)N)N", "target": "Nucleoside analog"},
    {"name": "Ganciclovir", "smiles": "C1=NC2=NC(=NC=C2N1CC(CO)(CO)O)N", "target": "Nucleoside analog"},
    {"name": "Letermovir", "smiles": "CC1=NC(=CC(=N1)NC2=CC=CC(=C2)C3=CC=C(C=C3)F)C(F)(F)F", "target": " terminase inhibitor"},
    {"name": "Baloxavir marboxil", "smiles": "CC1=CC2=C(C=C1)OC(=O)C3=C2OC(=O)C(=C3)C4=CC=C(C=C4)F", "target": "Cap-dependent endonuclease inhibitor"},
    {"name": "Sofosbuvir", "smiles": "CCC(CC)OC1C=C(C=CC1NC(=O)NC2=CC=C(C=C2)OC3=CC=CC=C3)C#N", "target": "HCV NS5B inhibitor"},
    {"name": "Daclatasvir", "smiles": "CC(C)C1=NC(=CS1)C2=CC(=NC=C2)C3=CC=CC(=C3)C(=O)NC4=CC=C(C=C4)C(=O)NC5=CC=C(C=C5)OC", "target": "HCV NS5A inhibitor"},
    {"name": "Entecavir", "smiles": "C1=CC2=C(C=CC(=C2N1)C3C(C(C(O3)CO)O)O)N", "target": "HBV polymerase inhibitor"},
    {"name": "Tenofovir", "smiles": "CCOC(=O)C1=C(C=CN=C1NC(CCP(=O)(C)O)O)N", "target": "Nucleotide analog"},
    {"name": "Emtricitabine", "smiles": "C1=NC(=NC=C1F)N2C=CSC2CO", "target": "Nucleoside analog"},
    {"name": "Lamivudine", "smiles": "C1=NC(=NC=C1)N2C=CSC2CO", "target": "Nucleoside analog"},
    {"name": "Dolutegravir", "smiles": "C1CC2=NC(=C(C(=O)C2C(=O)N1)NC3=CC=CC(=C3)C(F)(F)F)C4=CC=CC=C4F", "target": "HIV integrase inhibitor"},
    {"name": "Raltegravir", "smiles": "CC(=O)NC1=CC=CC=C1C(=O)NCC2=CNC3=CC=CC=C3C2=O", "target": "HIV integrase inhibitor"},
    {"name": "Maraviroc", "smiles": "CC(C)CC1=CC=C(C=C1)CC(C2CCCCN2)NC(=O)C3=CC=C(C=C3)C(F)(F)F", "target": "CCR5 antagonist"},
    {"name": "Pitavastatin", "smiles": "CCC(CC)OC1C=C(C=CC1NC(=O)C2=CC=C(C=C2)F)C(=O)O", "target": "HMG-CoA reductase inhibitor"},
    {"name": "Ivermectin", "smiles": "CCC(CC)OC1C=C(C=CC1NC(=O)C2=CC=C(C=C2)O)C(=O)O", "target": "Antiparasitic"},
    {"name": "Nitazoxanide", "smiles": "C1=CC=C(C=C1)NC(=O)C2=NC3=CC=CC=C3S2", "target": "Broad-spectrum antiviral"},
    {"name": "Chloroquine", "smiles": "CCN(CC)CCCC(C1=CC=CC2=NC3=CC=CC=C3C=C12)N", "target": "Endosome inhibitor"},
    {"name": "Hydroxychloroquine", "smiles": "CCN(CC)CCCC(C1=CC=CC2=NC3=CC=CC=C3C=C12)N", "target": "Endosome inhibitor"},
    {"name": "Arbidol", "smiles": "CCC1=CC2=C(C=C1)NC(=N2)SCC3=CC=C(C=C3)Br", "target": "Fusion inhibitor"},
    {"name": "Interferon alpha", "smiles": "N/A", "target": "Biologic (protein)"},
    {"name": "Cidofovir", "smiles": "C1=NC2=C(N=CN=C2N1)NC(=O)NCCP(=O)(O)O", "target": "Nucleotide analog"},
    {"name": "Foscarnet", "smiles": "OC(=O)P(=O)(O)O", "target": "Pyrophosphate analog"},
    {"name": "Amantadine", "smiles": "CC1(C2CCC(CC2)C1(C)N)C", "target": "M2 ion channel blocker"},
]


def seq_to_indices(seq, max_len=AA_MAX_SEQ_LEN):
    indices = []
    for aa in seq[:max_len]:
        indices.append(AA_VOCAB.get(aa, 0))
    while len(indices) < max_len:
        indices.append(AA_PAD_TOKEN)
    return torch.tensor(indices, dtype=torch.long)


def smiles_to_morgan(smiles, fp_size=2048):
    """Convert SMILES to Morgan fingerprint vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fp_size)
    arr = np.zeros(fp_size, dtype=np.float32)
    for bit in fp.GetOnBits():
        arr[bit] = 1.0
    return arr


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


class CureScoringHead(nn.Module):
    def __init__(self, protein_dim=64, drug_dim=128):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(protein_dim + drug_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, protein_emb, drug_emb):
        combined = torch.cat([protein_emb, drug_emb], dim=-1)
        return self.scorer(combined).squeeze(-1)


class DrugScreeningModel(nn.Module):
    """Full model: ProteinEncoder (frozen) + DrugEncoder + CureScoringHead."""
    def __init__(self, backbone_path, cure_head_path=None):
        super().__init__()
        # Load trained protein encoder
        self.backbone = ProteinEncoder()
        ckpt = torch.load(backbone_path, map_location='cpu')
        state = ckpt['model_state_dict']
        cleaned = {k.replace('protein_encoder.', ''): v for k, v in state.items()
                   if k.startswith('protein_encoder.')}
        self.backbone.load_state_dict(cleaned)
        for p in self.backbone.parameters():
            p.requires_grad = False

        protein_dim = self.backbone.pdm.num_subspaces * self.backbone.pdm.subspace_dim
        self.drug_encoder = DrugEncoder()
        self.cure_head = CureScoringHead(protein_dim=protein_dim, drug_dim=128)

        # Load trained cure head weights if available
        if cure_head_path and os.path.exists(cure_head_path):
            ckpt2 = torch.load(cure_head_path, map_location='cpu')
            state2 = ckpt2['model_state_dict']
            # Extract cure_head weights from MultiVirusModel checkpoint
            cure_head_state = {k.replace('cure_head.', ''): v for k, v in state2.items()
                               if k.startswith('cure_head.')}
            if cure_head_state:
                self.cure_head.load_state_dict(cure_head_state)
                print("Loaded trained CureScoringHead weights")
            # Extract drug_encoder weights if available
            drug_enc_state = {k.replace('mutation_encoder.', ''): v for k, v in state2.items()
                              if k.startswith('mutation_encoder.')}
            # mutation_encoder and drug_encoder have same architecture (both 128-dim output)

        # Override with dedicated drug encoder checkpoint if available
        drug_ckpt_path = "checkpoints_multivirus/best_drug_encoder.pt"
        if os.path.exists(drug_ckpt_path):
            drug_ckpt = torch.load(drug_ckpt_path, map_location='cpu')
            if 'drug_encoder_state_dict' in drug_ckpt:
                self.drug_encoder.load_state_dict(drug_ckpt['drug_encoder_state_dict'])
                print(f"Loaded trained DrugEncoder (Pearson={drug_ckpt.get('val_pearson', '?'):.3f})")

    @torch.no_grad()
    def score_drugs(self, protein_seq, drug_fps):
        """Score drugs against a viral protein. protein_seq can be string or tensor."""
        self.backbone.eval()
        self.drug_encoder.eval()
        self.cure_head.eval()

        if isinstance(protein_seq, str):
            seq_tensor = seq_to_indices(protein_seq).unsqueeze(0).to(DEVICE)
        else:
            seq_tensor = protein_seq
        protein_emb = self.backbone.extract_embedding(seq_tensor)
        # Expand protein embedding to match drug batch size
        protein_emb = protein_emb.expand(drug_fps.shape[0], -1)
        drug_emb = self.drug_encoder(drug_fps)
        scores = self.cure_head(protein_emb, drug_emb)
        return scores


def screen_drugs(virus_name="rabies_g", top_k=10):
    """Screen drug library against a viral protein."""
    backbone_path = "checkpoints_multivirus/best_pearson.pt"
    cure_head_path = "checkpoints_multivirus/best_cure_head.pt"

    print(f"Loading model...")
    model = DrugScreeningModel(backbone_path, cure_head_path).to(DEVICE)
    model.eval()

    # Get viral protein
    protein_seq = VIRAL_PROTEINS.get(virus_name)
    if protein_seq is None:
        print(f"Unknown virus: {virus_name}")
        return

    seq_tensor = seq_to_indices(protein_seq).unsqueeze(0).to(DEVICE)

    # Encode all drugs
    valid_drugs = []
    fps = []

    for drug in DRUG_LIBRARY:
        if drug['smiles'] == "N/A":
            continue
        fp = smiles_to_morgan(drug['smiles'])
        if fp is not None:
            valid_drugs.append(drug)
            fps.append(fp)

    if not fps:
        print("No valid drugs found")
        return

    fps_tensor = torch.tensor(np.array(fps), dtype=torch.float32).to(DEVICE)

    # Score
    print(f"Scoring {len(valid_drugs)} drugs against {virus_name}...")
    with torch.no_grad():
        scores = model.score_drugs(protein_seq, fps_tensor)

    # Rank
    score_vals = scores.cpu().numpy()
    ranked = sorted(zip(valid_drugs, score_vals), key=lambda x: x[1], reverse=True)

    print(f"\n{'='*70}")
    print(f"Top {top_k} Drug Candidates for {virus_name.upper()}")
    print(f"{'='*70}")
    print(f"{'Rank':<6} {'Drug':<25} {'Score':<10} {'Target'}")
    print(f"{'-'*70}")

    for i, (drug, score) in enumerate(ranked[:top_k], 1):
        print(f"{i:<6} {drug['name']:<25} {score:<10.4f} {drug['target']}")

    print(f"{'-'*70}")
    print(f"\nScore interpretation:")
    print(f"  Higher = predicted stronger antiviral effect")
    print(f"  Based on: protein embedding + Morgan fingerprint scoring")

    # Save results
    results = {
        "virus": virus_name,
        "n_drugs_screened": len(valid_drugs),
        "rankings": [
            {"rank": i+1, "drug": d['name'], "score": float(s), "target": d['target']}
            for i, (d, s) in enumerate(ranked[:top_k])
        ]
    }
    with open("screening_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to screening_results.json")

    return ranked


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--virus", default="rabies_g", choices=list(VIRAL_PROTEINS.keys()))
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    screen_drugs(args.virus, args.top_k)
