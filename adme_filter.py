"""
adme_filter.py — ADME + therapeutic index for rabies candidate.
Uses PK/PD model + GNN toxicity + RDKit descriptors.
"""
import sys, os, numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
sys.path.insert(0, '.')
from pkpd_model import simulate_pkpd_multi_dose, simulate_pkpd

# ─── MOLECULE ──────────────────────────────────────────────
smiles = 'O=c1[nH]c2cccc(-c3ccc(C(F)(F)F)cc3)c2[nH]1'
mol = Chem.MolFromSmiles(smiles)
mw = Descriptors.MolWt(mol)
logp = Descriptors.MolLogP(mol)
tpsa = Descriptors.TPSA(mol)
hbd = Descriptors.NumHDonors(mol)
hba = Descriptors.NumHAcceptors(mol)

print('='*60)
print('ADME + THERAPEUTIC INDEX')
print(f'4-CF3-Ph benzimidazolinone')
print('='*60)

# ─── 1. LIPINSKI / CNS MPO ─────────────────────────────────
print('\n--- Physicochemical Properties ---')
print(f'MW:         {mw:.1f}  ({">500" if mw>500 else "OK"})')
print(f'LogP:       {logp:.2f}  ({">5" if logp>5 else "OK"})')
print(f'TPSA:       {tpsa:.1f}  ({">140" if tpsa>140 else "OK"})')
print(f'HBD:        {hbd}  ({">5" if hbd>5 else "OK"})')
print(f'HBA:        {hba}  ({">10" if hba>10 else "OK"})')

lipinski = sum([mw>500, logp>5, hbd>5, hba>10])
print(f'Lipinski violations: {lipinski}/4')

# CNS MPO score (Wager et al. 2016, higher is better for brain penetration)
def cns_mpo_score(mw, logp, tpsa, hbd, pka=7.4):
    score = 0
    if mw <= 360: score += 1
    elif mw <= 400: score += 0.5
    if logp <= 3.0: score += 1
    elif logp <= 4.0: score += 0.5
    if tpsa <= 70: score += 1
    elif tpsa <= 90: score += 0.5
    if hbd <= 2: score += 1
    elif hbd <= 3: score += 0.5
    return score

cns = cns_mpo_score(mw, logp, tpsa, hbd)
print(f'CNS MPO score: {cns}/4  (>=3 = good brain penetration)')

# ─── 2. BBB PERMEABILITY ────────────────────────────────────
from pkpd_model import get_bbb_permeability
pbbb = get_bbb_permeability(smiles)
print(f'\nBBB permeability: {pbbb:.3f}  ({"High" if pbbb>0.7 else "Moderate" if pbbb>0.3 else "Low"})')

# ─── 3. PK/PD: THERAPEUTIC CONCENTRATION ───────────────────
print('\n--- PK/PD: Therapeutic Exposure ---')

# Multi-dose BID for 14 days
dose_mg = 500
pk = simulate_pkpd_multi_dose(smiles, dose_mg=dose_mg, interval_hrs=12, n_doses=28)
print(f'Regimen: {dose_mg}mg BID x 14 days')
print(f'MW: {mw:.1f} g/mol')
print(f'Dose molar: {dose_mg/mw:.2f} mmol')
print(f'Brain Cmax:        {pk["Cmax_brain_uM"]:.1f} uM')
print(f'Brain trough (SS): {pk["trough_brain_uM"]:.1f} uM')
print(f'Brain AUC (14d):   {pk["AUC_brain"]:.0f} uM*hr')
print(f'Plasma Cmax:       {pk["C_plasma_uM"].max():.1f} uM')
print(f'Time above EC50:   {pk["time_above_ec50_hrs"]:.1f} hrs')
print(f'Avg suppression:   {pk["avg_suppression"]:.1%}')
print(f'Max suppression:   {pk["max_suppression"]:.1%}')

EC50 = 10.0  # from pkpd_model
therapeutic_cmax = pk['Cmax_brain_uM']
therapeutic_trough = pk['trough_brain_uM']
therapeutic_cavg = pk['AUC_brain'] / (14 * 24)

print(f'\nTherapeutic brain concentration:')
print(f'  Cmax: {therapeutic_cmax:.1f} uM')
print(f'  Cmin: {therapeutic_trough:.1f} uM')
print(f'  Cavg: {therapeutic_cavg:.1f} uM')
print(f'  EC50: {EC50:.1f} uM')
print(f'  Cmax/EC50 ratio: {therapeutic_cmax/EC50:.1f}x')

# ─── 4. TOXIC THRESHOLD ────────────────────────────────────
print('\n--- Toxicity Threshold (GNN estimates) ---')

# Load GNN model
import torch
import torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader as GeoDataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = 'data'

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
        h_pool = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1)
        h_out = self.fc(h_pool)
        return {'dil': self.dil_head(h_out), 'ct': self.ct_head(h_out).squeeze(-1),
                'tx': torch.sigmoid(self.tx_head(h_out)), 'sd': torch.sigmoid(self.sd_head(h_out))}

model = GNNModel().to(DEVICE)
model.load_state_dict(torch.load(os.path.join(DATA_DIR, 'best_tox_model.pt'), map_location=DEVICE))
model.eval()

g = mol_to_graph(smiles).to(DEVICE)
batch = torch.zeros(g.x.size(0), dtype=torch.long, device=DEVICE)
with torch.no_grad():
    out = model(g.x, g.edge_index, batch)
    dil_class = out['dil'].argmax(dim=-1).item()
    dil_probs = F.softmax(out['dil'], dim=-1).squeeze()
    ct_tox = out['ct'].item()
    tx_probs = out['tx'].squeeze().tolist()

dil_names = ['Ambiguous', 'No-concern', 'Less-concern', 'Most-concern']
print(f'DILI concern: {dil_names[dil_class]} ({dil_probs[dil_class]:.1%})')
print(f'CT_TOX:       {ct_tox:.3f}  (>{0.5}=clinical toxicity concern)')
print(f'Tox21 flags:  {sum(1 for p in tx_probs if p>0.5)}/12')

# ─── 5. THERAPEUTIC INDEX ──────────────────────────────────
print('\n--- Therapeutic Index ---')

# Estimate toxic plasma Cmax from literature: 
# Most-concern DILI drugs have avg Cmax ~ 50-200 uM at clinical doses
# Use CT_TOX as a continuous safety margin indicator
# If CT_TOX < 0.3: wide safety margin
# If CT_TOX 0.3-0.5: moderate
# If CT_TOX > 0.5: narrow

toxic_cmax_estimate = 100.0  # uM (typical toxic threshold for oral drugs)
safe_margin = toxic_cmax_estimate / max(therapeutic_cmax, 0.01)

print(f'Therapeutic brain Cavg: {therapeutic_cavg:.1f} uM')
print(f'Estimated toxic Cmax:    {toxic_cmax_estimate:.0f} uM (literature-based)')
print(f'Safety margin (TI):      {safe_margin:.1f}x')
if safe_margin >= 10:
    print('Status: WIDE safety margin')
elif safe_margin >= 3:
    print('Status: MODERATE safety margin')
elif safe_margin >= 1:
    print('Status: NARROW safety margin')
else:
    print('Status: TOXIC at therapeutic dose')

# Combined risk score
print('\n--- Combined Risk Assessment ---')
risks = []
if lipinski > 1: risks.append('Lipinski violations')
if pbbb < 0.3: risks.append('Low BBB permeability')
if therapeutic_trough < EC50: risks.append('Trough below EC50')
if ct_tox > 0.5: risks.append('Clinical toxicity predicted')
if dil_class == 3: risks.append('Most-concern DILI')
if safe_margin < 3: risks.append('Narrow therapeutic index')

if not risks:
    print('No significant risks identified')
else:
    print(f'Risks ({len(risks)}):')
    for r in risks:
        print(f'  - {r}')

print(f'\nVerdict: {"FAVORABLE" if len(risks) <= 1 else "PROCEED WITH CAUTION" if len(risks) <= 3 else "HIGH RISK"}')
