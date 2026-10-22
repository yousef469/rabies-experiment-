"""
validate_withdrawn.py — Run pipeline on 10 known withdrawn drugs.
Pure inference — no training. Validates pipeline credibility.
"""
import sys, os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from rdkit import Chem
import warnings
warnings.filterwarnings('ignore')
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = 'data'

# ─── WITHDRAWN DRUGS ──────────────────────────────────────
withdrawn = [
    ('Troglitazone',  'CC1=C(C2=C(CCC(O2)(C)COC3=CC=C(C=C3)CC4C(=O)NC(=O)S4)C(=C1O)C)C',            'DILI'),
    ('Rofecoxib',     'CS(=O)(=O)C1=CC=C(C=C1)C2=C(C(=O)OC2)C3=CC=CC=C3',                             'Cardiovascular'),
    ('Cerivastatin',  'CC(C)C1=C(C(=C(C(=N1)C(C)C)COC)C2=CC=C(C=C2)F)/C=C/[C@H](C[C@H](CC(=O)O)O)O','Rhabdomyolysis'),
    ('Cisapride',     'CO[C@H]1CN(CC[C@H]1NC(=O)C2=CC(=C(C=C2OC)N)Cl)CCCOC3=CC=C(C=C3)F',             'Arrhythmia'),
    ('Terfenadine',   'CC(C)(C)C1=CC=C(C=C1)C(CCCN2CCC(CC2)C(C3=CC=CC=C3)(C4=CC=CC=C4)O)O',             'Arrhythmia'),
    ('Bromfenac',     'C1=CC(=C(C(=C1)C(=O)C2=CC=C(C=C2)Br)N)CC(=O)[O-].C1=CC(=C(C(=C1)C(=O)C2=CC=C(C=C2)Br)N)CC(=O)[O-].O.O.O.[Na+].[Na+]', 'DILI'),
    ('Benoxaprofen',  'CC(C1=CC2=C(C=C1)OC(=N2)C3=CC=C(C=C3)Cl)C(=O)O',                                'DILI'),
    ('Tolcapone',     'CC1=CC=C(C=C1)C(=O)C2=CC(=C(C(=C2)O)O)[N+](=O)[O-]',                           'DILI'),
    ('Pergolide',     'CCCN1C[C@@H](C[C@H]2[C@H]1CC3=CNC4=CC=CC2=C34)CSC.CS(=O)(=O)O',               'Heart valve'),
    ('Valdecoxib',    'CC1=C(C(=NO1)C2=CC=CC=C2)C3=CC=C(C=C3)S(=O)(=O)N',                             'Cardiovascular'),
]

# ─── MOL FEATURES ─────────────────────────────────────────
ATOMIC_NUMS = [6, 7, 8, 9, 16, 17, 35, 53, 15]

def atom_features(atom):
    atomic = [0]*len(ATOMIC_NUMS)
    if atom.GetAtomicNum() in ATOMIC_NUMS: atomic[ATOMIC_NUMS.index(atom.GetAtomicNum())] = 1
    else: atomic[0] = 1
    degree = atom.GetDegree(); deg = [0]*6
    if degree < 6: deg[degree] = 1
    chg = atom.GetFormalCharge()
    charge = [1 if chg == -1 else 0, 1 if chg == 0 else 0, 1 if chg == 1 else 0]
    num_h = atom.GetTotalNumHs(); h = [0]*5
    if num_h < 5: h[num_h] = 1
    hyb = atom.GetHybridization()
    hyb_map = {Chem.rdchem.HybridizationType.SP: 0, Chem.rdchem.HybridizationType.SP2: 1,
               Chem.rdchem.HybridizationType.SP3: 2, Chem.rdchem.HybridizationType.SP3D: 3,
               Chem.rdchem.HybridizationType.SP3D2: 4}
    hyb_vec = [0]*5
    if hyb in hyb_map: hyb_vec[hyb_map[hyb]] = 1
    aromatic = [int(atom.GetIsAromatic())]
    return atomic + deg + charge + h + hyb_vec + aromatic

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    node_feats = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.FloatTensor(node_feats)
    ei = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ei.append([i, j]); ei.append([j, i])
    if ei: edge_index = torch.LongTensor(ei).t().contiguous()
    else: edge_index = torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

# ─── MODEL ─────────────────────────────────────────────────
class GNNModel(nn.Module):
    def __init__(self, node_dim=29, hidden=64, num_layers=4):
        super().__init__()
        self.node_emb = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList(); self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINConv(mlp, train_eps=True))
            self.norms.append(nn.BatchNorm1d(hidden))
        self.fc = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Dropout(0.5))
        self.dil_head = nn.Linear(hidden, 4)
        self.ct_head = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        self.tx_head = nn.Linear(hidden, 12)
        self.sd_head = nn.Linear(hidden, 27)
    def forward(self, x, edge_index, batch):
        h = self.node_emb(x)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index); h = norm(h); h = F.relu(h)
        h = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1)
        h = self.fc(h)
        return {'dil': self.dil_head(h), 'ct': self.ct_head(h).squeeze(-1),
                'tx': torch.sigmoid(self.tx_head(h)), 'sd': torch.sigmoid(self.sd_head(h))}

model = GNNModel().to(DEVICE)
model_path = os.path.join(DATA_DIR, 'best_tox_model.pt')
if not os.path.exists(model_path):
    print(f'ERROR: {model_path} not found. Train first: train_toxicity.py')
    sys.exit(1)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()
print(f'Loaded model: {sum(p.numel() for p in model.parameters()):,} params')

# ─── PREDICT ───────────────────────────────────────────────
tx_endpoints = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
classes = ['Ambiguous', 'No-concern', 'Less-concern', 'Most-concern']

print(f'\n{"Drug":20} {"Reason":18} {"DILI":20} {"CT_TOX":>8} {"Tox21 flags":>14}')
print('-'*80)
for name, smiles, reason in withdrawn:
    g = mol_to_graph(smiles)
    if g is None:
        print(f'{name:20} SMILES PARSE FAILED')
        continue
    g = g.to(DEVICE)
    batch = torch.zeros(g.x.size(0), dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = model(g.x, g.edge_index, batch)
    dil_pred = out['dil'].argmax().item()
    ct_pred = out['ct'].item()
    tx_logits = out['tx'].squeeze().tolist()
    tx_flags = sum(1 for p in tx_logits if p > 0.5)
    probs = F.softmax(out['dil'], dim=-1).squeeze().tolist()
    print(f'{name:20} {reason:18} {classes[dil_pred]:20} {ct_pred:>8.3f} {tx_flags:>4}/12 ({max(probs):.0%})')

print()
print('Legend: Most-concern = pipeline correctly flags as dangerous')
print('Pipeline passes if: most withdrawn drugs flagged Most-concern or have CT_TOX>0.5 or 4+ Tox21 flags')
