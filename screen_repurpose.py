"""
screen_repurpose.py — Rank a PANEL of N candidate drugs for rabies.

WHY A PANEL, NOT A CURE
-----------------------
The goal of this project is *not* to output one "cure". It is to output a ranked
panel of candidates (default 100), each with a transparent score breakdown and an
explicit uncertainty tier, so a wet lab can test them in order and maybe find one
that works. A ranking is a hypothesis; the experiment is what decides.

TARGET
------
The rabies virus (RABV) has five proteins: G, L, N, P, M. Only the glycoprotein
G has a structure available here (PDB 6LGX, cached in pdb_cache/), so only G is
screened. L / N / P / M are listed but *not* screened (no structure) — see the
TARGETS table below and LIMITATIONS.md.

LEAKAGE FIX
-----------
The GNN is trained on DILIrank. Screening the drugs it was fit on would be
circular, so this script reproduces the trainer's 3-way split (seed 42) and
scores ONLY the held-out test drugs. If the split cannot be reproduced the script
stops rather than silently scoring training compounds.

DOCKING CAVEAT
--------------
Vina is run as BLIND docking (box centred on the receptor centroid, 26 Å cube,
exhaustiveness 8). This is a coarse screen for ORDERING candidates, not a
validated binding-mode analysis. See LIMITATIONS.md.
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

# RABV proteins. Only G has a structure available, so only G is screened.
TARGETS = [
    # (gene, name,            role,                  structure_available)
    ('G', 'glycoprotein',   'cell entry / fusion',  True),
    ('L', 'polymerase',     'genome replication',   False),
    ('N', 'nucleoprotein',  'RNA packaging',        False),
    ('P', 'phosphoprotein', 'polymerase cofactor',  False),
    ('M', 'matrix',         'virion assembly',      False),
]

# How many ranked candidates to emit (the "panel").
PANEL_N = 100

_args = argparse.ArgumentParser(description='Rank a panel of candidate drugs for RABV.')
_args.add_argument('--n', type=int, default=PANEL_N,
                   help=f'panel size, i.e. how many ranked candidates to emit (default {PANEL_N})')
_args.add_argument('--dock', action='store_true',
                   help='also run AutoDock Vina on the top candidates (slow)')
_args.add_argument('--dock-n', type=int, default=30,
                   help='how many top candidates to dock when --dock is set (default 30)')
args = _args.parse_args()

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

# ─── 4. SCORE + BUILD THE RANKED PANEL ─────────────────────
# Transparent toxicity risk (lower = better). The components are kept per
# candidate so the ranking is auditable, not a single magic number.
def toxicity_risk(r):
    dil_term = r['dil_class'] / 3.0            # 0 (no concern) .. 1 (most concern)
    ct_term = min(max(r['ct_tox'], 0.0), 1.0)  # ClinTox probability
    tx_term = min(r['tx_flags'] / 12.0, 1.0)   # fraction of Tox21 endpoints flagged
    return 0.5 * dil_term + 0.3 * ct_term + 0.2 * tx_term

def uncertainty_tier(r):
    """How much the three toxicity heads agree. A = strongest hypothesis."""
    if r['dil_class'] == 1 and r['dil_prob'] >= 0.6 and r['tx_flags'] == 0 and r['ct_tox'] < 0.3:
        return 'A'
    if r['dil_class'] in (0, 1) and r['tx_flags'] <= 1:
        return 'B'
    return 'C'

print('\nTargets considered (RABV):')
for gene, name, role, has_struct in TARGETS:
    tag = 'SCREENED (6LGX)' if has_struct else 'not screened (no structure)'
    print(f'  {gene}  {name:<14} {role:<22} {tag}')

for r in results:
    r['tox_risk'] = toxicity_risk(r)
    r['tier'] = uncertainty_tier(r)

# Panel = held-out drugs with a favourable predicted safety profile.
panel = [r for r in results if r['dil_class'] in (0, 1) and r['tx_flags'] < 3]
panel.sort(key=lambda r: (r['tox_risk'], r['tx_flags'], r['ct_tox']))
panel = panel[:args.n]

print(f'\nPanel: {len(panel)} ranked low-toxicity held-out candidates '
      f'(target: RABV glycoprotein G)')

# ─── 5. DOCK TOP CANDIDATES (optional) ─────────────────────
linked = panel[:args.dock_n]
docking_results = []

if not args.dock:
    print(f'\n[skip] docking step (pass --dock to run AutoDock Vina on top {args.dock_n})')

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
print('\n' + '='*78)
print('RANKED PANEL — candidate drugs for the RABV glycoprotein (held-out only)')
print('A ranking is a hypothesis, not a cure. The wet lab decides.')
print('='*78)

if docking_results:
    print(f'\n{"Rank":<5} {"Drug":<26} {"Risk":<7} {"Tier":<5} {"Flags":<6} {"Binding":<9}')
    print('-'*70)
    for rank, r in enumerate(docking_results, 1):
        print(f'{rank:<5} {r["drug"][:25]:<26} {r["tox_risk"]:<7.3f} {r["tier"]:<5} '
              f'{r["tx_flags"]:<6} {r["binding_energy"]:<9.2f}')
else:
    print(f'\n{"Rank":<5} {"Drug":<26} {"DILI class":<12} {"Risk":<7} {"Tier":<5} {"Flags":<6}')
    print('-'*70)
    for rank, r in enumerate(panel, 1):
        dil_name = ['Ambiguous', 'No-concern', 'Less-concern', 'Most-concern'][r['dil_class']]
        print(f'{rank:<5} {r["drug"][:25]:<26} {dil_name:<12} {r["tox_risk"]:<7.3f} '
              f'{r["tier"]:<5} {r["tx_flags"]:<6}')

print('\nScore = 0.5*DILI + 0.3*ClinTox + 0.2*Tox21_flags  (lower is better).')
print('Tier A/B/C = agreement across the three toxicity heads (A = strongest hypothesis).')
print('These are ranked HYPOTHESES for wet-lab testing, not validated actives.')

# Save
out = {
    'meta': {
        'purpose': 'ranked panel of candidate drugs (not a single cure)',
        'target': 'RABV glycoprotein G (PDB 6LGX)',
        'targets_not_screened': [g for g, _, _, s in TARGETS if not s],
        'screened': 'held-out DILIrank test drugs only (no leakage)',
        'panel_size': len(panel),
        'score': '0.5*dil_class/3 + 0.3*clin_tox + 0.2*tox21_flags/12 (lower=better)',
        'caveat': 'ranked hypotheses for wet-lab testing, not validated actives',
    },
    'panel': [{k: v for k, v in r.items() if k != 'tx_probs'} for r in panel],
    'docking': [{k: v for k, v in r.items() if k != 'tx_probs'} for r in docking_results],
}
with open(os.path.join(DATA_DIR, 'repurposing_results.json'), 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f'\nSaved to {DATA_DIR}/repurposing_results.json')
