"""
gnn_explain.py — Gradient saliency for Tox21 endpoint structural alerts.
Computes gradient of each endpoint output w.r.t. node features to identify
which atoms drive each prediction.
"""
import sys, os, json, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from rdkit import Chem
import warnings
warnings.filterwarnings('ignore')

from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = 'data'

TX_ENDPOINTS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

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
        return None, None
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
    return Data(x=x, edge_index=edge_index), mol

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
        return self.tx_head(h_out)

base_model = GNNModel().to(DEVICE)
base_model.load_state_dict(torch.load(os.path.join(DATA_DIR, 'best_tox_model.pt'), map_location=DEVICE))
base_model.eval()

def analyze(name, smiles):
    print(f'\n{"="*60}')
    print(f'{name}')
    print(f'SMILES: {smiles}')
    print('='*60)

    graph, mol = mol_to_graph(smiles)
    if graph is None:
        print('  FAILED')
        return {}

    graph = graph.to(DEVICE)
    batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=DEVICE)

    # Predict
    with torch.no_grad():
        tx_logits = base_model(graph.x, graph.edge_index, batch)
        tx_probs = torch.sigmoid(tx_logits).squeeze().tolist()
        full = GNNModel().to(DEVICE)
        full.load_state_dict(torch.load(os.path.join(DATA_DIR, 'best_tox_model.pt'), map_location=DEVICE))
        full.eval()
        h = full.node_emb(graph.x)
        for conv, norm in zip(full.convs, full.norms):
            h = conv(h, graph.edge_index)
            h = norm(h)
            h = F.relu(h)
        h_pool = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1)
        h_out = full.fc(h_pool)
        dil_probs = F.softmax(full.dil_head(h_out), dim=-1)
        dil_class = dil_probs.argmax(dim=-1).item()
        dil_name = ['Ambiguous', 'No-concern', 'Less-concern', 'Most-concern'][dil_class]

    print(f'  DILI: {dil_name} ({dil_probs[0, dil_class]:.1%})')
    flagged = []
    for i, ep in enumerate(TX_ENDPOINTS):
        flag = ' ***' if tx_probs[i] > 0.5 else ''
        print(f'    {ep:15}: {tx_probs[i]:.3f}{flag}')
        if tx_probs[i] > 0.5:
            flagged.append(i)

    if not flagged:
        print('  (no flagged endpoints)')
        return {}

    # Gradient saliency for each flagged endpoint
    explanations = {}
    for ep_idx in flagged:
        ep_name = TX_ENDPOINTS[ep_idx]

        x = graph.x.clone().detach().requires_grad_(True)
        logits = base_model(x, graph.edge_index, batch)
        base_model.zero_grad()
        logits[:, ep_idx].backward()
        saliency = x.grad.norm(dim=-1).cpu().numpy()
        saliency = saliency / saliency.max()

        top_k = min(5, len(saliency))
        top_idx = np.argsort(saliency)[-top_k:][::-1]

        print(f'\n  >>> {ep_name} <<<')
        print(f'    Top {top_k} atoms:')
        for rank, aidx in enumerate(top_idx):
            atom = mol.GetAtomWithIdx(int(aidx))
            neighbors = [mol.GetAtomWithIdx(n.GetIdx()).GetSymbol() for n in atom.GetNeighbors()]
            print(f'      {rank+1}. {atom.GetSymbol()}{int(aidx)} ({",".join(neighbors)}) — sal={saliency[aidx]:.3f}')

        explanations[ep_name] = {
            'saliency': saliency.tolist(),
            'top_atoms': [int(a) for a in top_idx],
            'top_symbols': [mol.GetAtomWithIdx(int(a)).GetSymbol() for a in top_idx],
        }

    return explanations

# ─── RUN ────────────────────────────────────────────────────
results = {}

results['rabies_candidate'] = analyze(
    'Rabies candidate (4-CF3-Ph benzimidazolinone)',
    'O=c1[nH]c2cccc(-c3ccc(C(F)(F)F)cc3)c2[nH]1',
)

hits = [
    ('Ripretinib', 'CC(C)(C)NC1=NC(=NC(=C1F)C2=CC3=C(C=C2)N=CN3C4CC4)N5CCOCC5'),
    ('Pralsetinib', 'CC1(COC2=C1C(=NC(=N2)NC3=NC=CC(=C3)OC4=C(C=CC=C4F)F)NC5=NC=CC(=N5)C#N)C'),
    ('Selpercatinib', 'CC1(CCC2=C(C1)N(C(=N2)C3=CC(=C(C=C3)F)OC4=CC5=C(C=C4)N(C(=O)C=C5)C)O)C'),
    ('Duvelisib', 'CN1C=C(C2=CC(=NC=C2)NC3=NC=CC(=N3)C4=C(C=CC=C4Cl)F)C5=CC=CC=C15'),
    ('Belumosudil', 'CN1CCN(CC1)C2=C3C=C(SC3=NC(=N2)N4CCN(CC4)C(=O)C(C)(C)O)C#N'),
]

for drug_name, drug_smiles in hits:
    results[drug_name] = analyze(drug_name, drug_smiles)

with open(os.path.join(DATA_DIR, 'gnn_explanations.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f'\nSaved {len(results)} molecule analyses to data/gnn_explanations.json')
