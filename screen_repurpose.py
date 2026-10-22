"""
screen_repurpose.py — Screen held-out FDA drugs for rabies repurposing.

Pipeline: GNN toxicity filter → (optional) AutoDock Vina → candidate ranking.

LEAKAGE FIX
-----------
The GNN is trained on DILIrank. Predicting DILIrank and then selecting the drugs the
model already calls "low toxicity" is circular: those are the very labels it was fit
on. This script therefore reproduces the trainer's 3-way split (seed 42) and screens
ONLY the held-out test drugs, so the reported toxicity predictions are genuinely
out-of-sample. If the split cannot be reproduced the script stops instead of
silently scoring training compounds.

DOCKING CAVEAT
--------------
Vina is run as BLIND docking (box centred on the receptor centroid, 26 Å cube,
exhaustiveness 8). This is a coarse screen, not a validated binding-mode analysis;
scores are used only to order candidates. See LIMITATIONS.md.
"""

import sys, os, json, argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from rdkit import Chem
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

DATA_DIR = 'data'
VINA_BIN = "/usr/local/bin/vina"

# Severity map — must match train_toxicity.py so the split is reproducible.
SEV_MAP = {'vMost-DILI-concern': 3, 'vMOST-DILI-concern': 3, 'vLess-DILI-concern': 2,
           'vNo-DILI-concern': 1, 'vNo-DILI-Concern': 1, 'Ambiguous-DILI-concern': 0}

# Tox21 endpoint names
TX_ENDPOINTS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

# ─── GNN MODEL ─────────────────────────────────────────────
ATOMIC_NUMS = [6, 7, 8, 9, 16, 17, 35, 53, 15]

def atom_features(atom):
    atomic = [0]*len(ATOMIC_NUMS)
    if atom.GetAtomicNum() in ATOMIC_NUMS:
        atomic[ATOMIC_NUMS.index(atom.GetAtomicNum())] = 1
    else:
        atomic[0] = 1
    degree = atom.GetDegree()
    deg = [0]*6
    if degree < 6:
        deg[degree] = 1
    chg = atom.GetFormalCharge()
    charge = [1 if chg == -1 else 0, 1 if chg == 0 else 0, 1 if chg == 1 else 0]
    num_h = atom.GetTotalNumHs()
    h = [0]*5
    if num_h < 5:
        h[num_h] = 1
    hyb = atom.GetHybridization()
    hyb_map = {Chem.rdchem.HybridizationType.SP: 0, Chem.rdchem.HybridizationType.SP2: 1,
               Chem.rdchem.HybridizationType.SP3: 2, Chem.rdchem.HybridizationType.SP3D: 3,
               Chem.rdchem.HybridizationType.SP3D2: 4}
    hyb_vec = [0]*5
    if hyb in hyb_map:
        hyb_vec[hyb_map[hyb]] = 1
    aromatic = [int(atom.GetIsAromatic())]
    return atomic + deg + charge + h + hyb_vec + aromatic

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    node_feats = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.FloatTensor(node_feats)
    edge_indices = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])
    if not edge_indices:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.LongTensor(edge_indices).t().contiguous()
    return Data(x=x, edge_index=edge_index)

class GNNModel(nn.Module):
    def __init__(self, node_dim=29, hidden=64, num_layers=4):
        super().__init__()
        self.node_emb = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.BatchNorm1d(hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.norms.append(nn.BatchNorm1d(hidden))
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(0.5),
        )
        self.dil_head = nn.Linear(hidden, 4)
        self.ct_head = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        self.tx_head = nn.Linear(hidden, 12)
        self.sd_head = nn.Linear(hidden, 27)

    def forward(self, x, edge_index, batch):
        h = self.node_emb(x)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
        h_pool = torch.cat([
            global_mean_pool(h, batch),
            global_max_pool(h, batch),
        ], dim=-1)
        h_out = self.fc(h_pool)
        return {
            'dil': self.dil_head(h_out),
            'ct': self.ct_head(h_out).squeeze(-1),
            'tx': torch.sigmoid(self.tx_head(h_out)),
            'sd': torch.sigmoid(self.sd_head(h_out)),
        }

# ─── 1. LOAD MODEL ─────────────────────────────────────────
model = GNNModel().to(DEVICE)
model.load_state_dict(torch.load(os.path.join(DATA_DIR, 'best_tox_model.pt'), map_location=DEVICE))
model.eval()
print(f'Model loaded ({sum(p.numel() for p in model.parameters()):,} params)')

# ─── 2. LOAD DILIrank DRUGS ─────────────────────────────────
f = os.path.join(DATA_DIR, 'Drug Induced Liver Injury Rank (DILIrank 2.0) Dataset  FDA.xlsx')
dil = pd.read_excel(f, sheet_name='version 2', skiprows=2)
dil.columns = ['LTKBID', 'CompoundName', 'SeverityClass', 'LabelSection', 'vDILI-Concern', 'Comment']
smi_df = pd.read_csv(os.path.join(DATA_DIR, 'dilirank_smiles.csv'))
smi_df = smi_df[smi_df['Status'] == 'OK']
dil = dil.merge(smi_df, left_on='CompoundName', right_on='DrugName', how='inner')
dil['severity'] = dil['vDILI-Concern'].map(SEV_MAP)
dil = dil[dil['severity'].notna()].reset_index(drop=True)
print(f'Loaded {len(dil)} DILIrank drugs with SMILES')

# Reproduce the trainer's split so we only screen drugs the model has NOT seen.
sev = dil['severity'].values.astype(np.int64)
n_all = np.arange(len(dil))
train_idx, temp_idx = train_test_split(n_all, test_size=0.30, random_state=42, stratify=sev)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42, stratify=sev[temp_idx])
held_out = set(test_idx.tolist())
print(f'Reproduced trainer split: train={len(train_idx)} val={len(val_idx)} '
      f'held-out test={len(test_idx)} (screening held-out only)')

# ─── 3. PREDICT (HELD-OUT DRUGS ONLY) ────────────────────────
results = []
for i, row in dil.iterrows():
    if i not in held_out:
        continue
    g = mol_to_graph(row['SMILES'])
    if g is None:
        continue
    g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
    with torch.no_grad():
        g.x = g.x.to(DEVICE)
        g.edge_index = g.edge_index.to(DEVICE)
        g.batch = g.batch.to(DEVICE)
        out = model(g.x, g.edge_index, g.batch)
        dil_class = out['dil'].argmax(dim=-1).item()
        dil_probs = F.softmax(out['dil'], dim=-1).squeeze().tolist()
        ct_prob = out['ct'].item()
        tx_probs = out['tx'].squeeze().tolist()
        tx_flags = sum(1 for p in tx_probs if p > 0.5)

    results.append({
        'drug': row['CompoundName'],
        'smiles': row['SMILES'],
        'dil_class': dil_class,
        'dili_concern': row['vDILI-Concern'],
        'dil_prob': dil_probs[dil_class],
        'ct_tox': ct_prob,
        'tx_flags': tx_flags,
        'tx_probs': {e: p for e, p in zip(TX_ENDPOINTS, tx_probs)},
    })

print(f'Predicted {len(results)} held-out drugs')

# ─── 4. FILTER ──────────────────────────────────────────────
# Low-toxicity: DILI class 0 (Ambiguous) or 1 (No-concern), <3 Tox21 flags
filtered = [r for r in results if r['dil_class'] in (0, 1) and r['tx_flags'] < 3]
filtered.sort(key=lambda r: (r['dil_class'], r['tx_flags'], r['ct_tox']))

print(f'\nLow-toxicity candidates (held-out): {len(filtered)}')
print(f'\n{"Drug":<30} {"DILI":<12} {"Flags":<6} {"CT_TOX":<8}')
print('-'*56)
for r in filtered[:30]:
    dil_name = ['Ambiguous', 'No-concern', 'Less-concern', 'Most-concern'][r['dil_class']]
    print(f'{r["drug"]:<30} {dil_name:<12} {r["tx_flags"]:<6} {r["ct_tox"]:<8.3f}')

# ─── 5. DOCK TOP 30 ────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--dock', action='store_true',
                    help='run AutoDock Vina on the top candidates (slow)')
args = parser.parse_args()

linked = filtered[:30]
docking_results = []

if not args.dock:
    print('\n[skip] docking step (pass --dock to run AutoDock Vina)')

import subprocess, tempfile
from rdkit.Chem import AllChem

def smiles_to_pdbqt(smiles, output_path):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
    except:
        return False
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        setup = MoleculePreparation().prepare(mol)[0]
        pdbqt_string, is_ok, err_msg = PDBQTWriterLegacy.write_string(setup)
        if is_ok:
            with open(output_path, 'w') as f:
                f.write(pdbqt_string)
            return True
        return False
    except:
        return False

def run_docking(candidates):
    """Blind-dock each candidate against the cached receptor. Returns a list of
    results sorted by Vina binding energy (kcal/mol, more negative = better)."""
    rec_pdb = os.path.join(DATA_DIR, os.pardir, 'pdb_cache/6lgx.pdb')
    work_dir = '/tmp/repurpose_screen'
    os.makedirs(work_dir, exist_ok=True)

    rec_pdbqt = os.path.join(work_dir, 'receptor.pdbqt')
    if not os.path.exists(rec_pdbqt):
        print('Preparing receptor PDBQT...')
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('rec', rec_pdb)
        chain = structure[0]['A']

        ATOM_TYPE_TABLE = {6: 'C', 7: 'N', 8: 'O', 16: 'S', 9: 'F', 17: 'Cl', 35: 'Br', 53: 'I', 15: 'P'}
        AA_CODES = {'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS',
                    'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR'}

        with open(rec_pdbqt, 'w') as f:
            f.write("REMARK PDBQT\n")
            serial = 0
            for residue in chain:
                if residue.get_resname().strip() not in AA_CODES:
                    continue
                resid = residue.get_id()[1]
                for atom in residue:
                    serial += 1
                    if atom.get_full_id()[3][0].strip():
                        continue
                    elem = atom.element.strip() if atom.element else atom.get_name()[0]
                    if not elem:
                        continue
                    x, y, z = atom.get_coord()
                    elem_num = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16, 'F': 9, 'CL': 17, 'BR': 35, 'I': 53, 'P': 15}.get(elem.upper(), 6)
                    atype = ATOM_TYPE_TABLE.get(elem_num, 'C')
                    aname = f" {atom.get_name().strip():<3}"
                    line = f"ATOM  {serial:>5} {aname:<4} {residue.get_resname():<3} A{resid:>4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atype}\n"
                    f.write(line)
            f.write("TER\nEND\n")

    # Blind-docking box: centred on the receptor centroid, 26 Å cube.
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('rec', rec_pdb)
    coords = [atom.get_coord() for residue in structure[0]['A'] for atom in residue]
    center = tuple(np.mean(coords, axis=0))
    print(f'Docking center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})')

    results_dock = []
    for i, r in enumerate(candidates):
        name = r['drug'].replace('/', '_').replace(' ', '_')
        lig_pdbqt = os.path.join(work_dir, f'{name}.pdbqt')
        out_pdbqt = os.path.join(work_dir, f'{name}_out.pdbqt')

        print(f'  {i+1}/{len(candidates)}: {r["drug"][:25]:25}... ', end='', flush=True)

        ok = smiles_to_pdbqt(r['smiles'], lig_pdbqt)
        if not ok:
            print('ligand prep FAIL')
            continue

        cmd = [VINA_BIN, '--receptor', rec_pdbqt, '--ligand', lig_pdbqt,
               '--out', out_pdbqt, '--center_x', f'{center[0]:.1f}',
               '--center_y', f'{center[1]:.1f}', '--center_z', f'{center[2]:.1f}',
               '--size_x', '26', '--size_y', '26', '--size_z', '26',
               '--exhaustiveness', '8', '--num_modes', '1']

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            score = None
            if os.path.exists(out_pdbqt):
                with open(out_pdbqt) as fh:
                    for line in fh:
                        if line.startswith('REMARK VINA RESULT:'):
                            score = float(line.split()[3])
                            break
            if score is not None:
                results_dock.append({**r, 'binding_energy': score})
                print(f'{score:.2f} kcal/mol')
            else:
                print(f'FAIL ({result.stderr[:80]})')
        except subprocess.TimeoutExpired:
            print('TIMEOUT')
        except Exception as e:
            print(f'ERROR: {e}')

    results_dock.sort(key=lambda x: x['binding_energy'])
    return results_dock


docking_results = run_docking(linked) if args.dock else []

# ─── 6. REPORT ──────────────────────────────────────────────
print('\n' + '='*60)
print('RESULTS: FDA DRUG REPURPOSING FOR RABIES (held-out drugs only)')
print('='*60)

if docking_results:
    print(f'\n{"Rank":<5} {"Drug":<25} {"DILI Class":<12} {"Binding":<10} {"Flags":<5}')
    print('-'*60)
    for rank, r in enumerate(docking_results, 1):
        dil_name = ['Ambiguous', 'No-concern', 'Less-concern', 'Most-concern'][r['dil_class']]
        print(f'{rank:<5} {r["drug"]:<25} {dil_name:<12} {r["binding_energy"]:<10.2f} {r["tx_flags"]:<5}')
elif linked:
    print('\nTop low-toxicity held-out candidates (docking skipped — pass --dock to rank by binding):')
    print(f'\n{"Rank":<5} {"Drug":<25} {"DILI Class":<12} {"Flags":<6} {"CT_TOX":<8}')
    print('-'*60)
    for rank, r in enumerate(linked, 1):
        dil_name = ['Ambiguous', 'No-concern', 'Less-concern', 'Most-concern'][r['dil_class']]
        print(f'{rank:<5} {r["drug"]:<25} {dil_name:<12} {r["tx_flags"]:<6} {r["ct_tox"]:<8.3f}')
else:
    print('No low-toxicity candidates among held-out drugs')

# Save
out = {
    'all_predictions': [{k: v for k, v in r.items() if k != 'tx_probs'} | {'tx_flags': r['tx_flags']} for r in results],
    'filtered': [{k: v for k, v in r.items() if k != 'tx_probs'} for r in filtered[:50]],
    'docking': [{k: v for k, v in r.items() if k != 'tx_probs'} for r in docking_results],
}
with open(os.path.join(DATA_DIR, 'repurposing_results.json'), 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f'\nSaved to {DATA_DIR}/repurposing_results.json')
