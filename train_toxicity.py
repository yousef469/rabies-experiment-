"""
train_toxicity.py — GNN backbone for molecular graph learning.
GINConv message passing + global mean/max pooling.
Multi-task: DILIrank + Tox21 + ClinTox + SIDER.
"""
import sys, os, json, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from rdkit import Chem
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 200
EARLY_STOP = 30
DATA_DIR = 'data'

# ─── 1. LOAD ─────────────────────────────────────────────────
def load_all_data():
    f = os.path.join(DATA_DIR, 'Drug Induced Liver Injury Rank (DILIrank 2.0) Dataset  FDA.xlsx')
    dil = pd.read_excel(f, sheet_name='version 2', skiprows=2)
    dil.columns = ['LTKBID', 'CompoundName', 'SeverityClass', 'LabelSection', 'vDILI-Concern', 'Comment']
    smi_df = pd.read_csv(os.path.join(DATA_DIR, 'dilirank_smiles.csv'))
    smi_df = smi_df[smi_df['Status'] == 'OK']
    dil = dil.merge(smi_df, left_on='CompoundName', right_on='DrugName', how='inner')
    sev_map = {'vMost-DILI-concern': 3, 'vMOST-DILI-concern': 3, 'vLess-DILI-concern': 2,
               'vNo-DILI-concern': 1, 'vNo-DILI-Concern': 1, 'Ambiguous-DILI-concern': 0}
    dil['severity'] = dil['vDILI-Concern'].map(sev_map)
    dil = dil[dil['severity'].notna()]
    ct = pd.read_csv(os.path.join(DATA_DIR, 'clintox.csv'))
    tx = pd.read_csv(os.path.join(DATA_DIR, 'tox21.csv'))
    sd = pd.read_csv(os.path.join(DATA_DIR, 'sider.csv'))
    return dil, ct, tx, sd

dil, ct, tx, sd = load_all_data()
print(f'DILIrank: {len(dil)}, ClinTox: {len(ct)}, Tox21: {len(tx)}, SIDER: {len(sd)}')

# ─── 2. GRAPH FEATURES ──────────────────────────────────────
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

def process_graphs(df, smiles_col='SMILES'):
    graphs, idxs = [], []
    for i, row in df.iterrows():
        g = mol_to_graph(row[smiles_col])
        if g is not None:
            graphs.append(g)
            idxs.append(i)
    return graphs, idxs

dil_graphs, dil_idx = process_graphs(dil)
ct_graphs, ct_idx = process_graphs(ct, 'smiles')
tx_graphs, tx_idx = process_graphs(tx, 'smiles')
sd_graphs, sd_idx = process_graphs(sd, 'smiles')
print(f'Graphs: DILIrank={len(dil_graphs)}, ClinTox={len(ct_graphs)}, Tox21={len(tx_graphs)}, SIDER={len(sd_graphs)}')

# Labels
dil_y = dil.iloc[dil_idx]['severity'].values.astype(np.int64)
ct_y = ct.iloc[ct_idx]['CT_TOX'].values.astype(np.float32)
tx_cols = [c for c in tx.columns if c not in ['mol_id', 'smiles']]
tx_y = tx.iloc[tx_idx][tx_cols].values.astype(np.float32)
tx_y = np.nan_to_num(tx_y, nan=-1)
sd_cols = [c for c in sd.columns if c != 'smiles']
sd_y = sd.iloc[sd_idx][sd_cols].values.astype(np.float32)
sd_y = np.nan_to_num(sd_y, nan=-1)

# ─── 3. GNN MODEL ───────────────────────────────────────────
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

# ─── 4. MULTI-TASK TRAINING ────────────────────────────────
# 3-way split. The model is selected on val and the reported number is measured on
# the held-out TEST set. NOTE: this is a stratified RANDOM split, so structurally
# similar molecules can appear in both train and test — it is an OPTIMISTIC upper
# bound. For the honest, unseen-scaffold estimate run:  python benchmark.py --split scaffold
n_all = np.arange(len(dil_graphs))
train_idx, temp_idx = train_test_split(n_all, test_size=0.30, random_state=42, stratify=dil_y)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42, stratify=dil_y[temp_idx])

# Inverse-frequency class weights, padded to all 4 DILI classes (a class may be
# absent from a small training split).
classes, counts = np.unique(dil_y[train_idx], return_counts=True)
weights = counts.sum() / (len(classes) * counts.astype(float))
cw = torch.ones(4)
for c, w in zip(classes, weights):
    cw[c] = w
cw = cw.to(DEVICE)
print(f'Class weights: {dict(zip(classes.tolist(), np.round(weights, 3).tolist()))}')
print(f'Split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}')

# Assign task-specific labels to each graph
for g, y in zip(dil_graphs, dil_y):
    g.y = torch.LongTensor([y])
for g, y in zip(ct_graphs, ct_y):
    g.y = torch.FloatTensor([y])
for g, y in zip(tx_graphs, tx_y):
    g.y = torch.FloatTensor(y)
for g, y in zip(sd_graphs, sd_y):
    g.y = torch.FloatTensor(y)

train_dil = [dil_graphs[i] for i in train_idx]
val_dil = [dil_graphs[i] for i in val_idx]
test_dil = [dil_graphs[i] for i in test_idx]

loaders = {
    'dil': GeoDataLoader(train_dil, BATCH_SIZE, shuffle=True),
    'ct': GeoDataLoader(ct_graphs, BATCH_SIZE, shuffle=True),
    'tx': GeoDataLoader(tx_graphs, BATCH_SIZE, shuffle=True),
    'sd': GeoDataLoader(sd_graphs, BATCH_SIZE, shuffle=True),
}
val_loader = GeoDataLoader(val_dil, BATCH_SIZE)
test_loader = GeoDataLoader(test_dil, BATCH_SIZE)

model = GNNModel().to(DEVICE)
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)

best_val_acc = 0
patience = 0
task_order = ['dil', 'ct', 'tx', 'sd']
task_weight = {'dil': 2.0, 'ct': 1.0, 'tx': 1.0, 'sd': 0.5}

print('\nMulti-task training (DILIrank + ClinTox + Tox21 + SIDER)...')
for epoch in range(EPOCHS):
    model.train()
    iters = {t: iter(l) for t, l in loaders.items()}
    steps = max(len(train_dil) // BATCH_SIZE, 1)
    train_loss = 0

    for _ in range(steps):
        total_loss = 0
        for task in task_order:
            try:
                batch = next(iters[task])
            except StopIteration:
                iters[task] = iter(loaders[task])
                batch = next(iters[task])
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.batch)

            if task == 'dil':
                loss = F.cross_entropy(out['dil'], batch.y.view(-1), weight=cw)
            elif task == 'ct':
                loss = F.binary_cross_entropy(out['ct'], batch.y.view(-1))
            elif task == 'tx':
                y = batch.y.view(-1, 12)
                mask = y != -1
                loss = F.binary_cross_entropy(out['tx'], y.clamp(0, 1), reduction='none')
                loss = (loss * mask).sum() / mask.sum().clamp(min=1)
            elif task == 'sd':
                y = batch.y.view(-1, 27)
                loss = F.binary_cross_entropy(out['sd'], y)

            total_loss = total_loss + loss * task_weight[task]

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += total_loss.item()

    # Validate on DILIrank
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out['dil'], batch.y.view(-1))
            val_loss += loss.item()
            preds = out['dil'].argmax(dim=-1)
            correct += (preds == batch.y.view(-1)).sum().item()
            total += len(batch.y)

    val_acc = correct / total
    val_loss /= len(val_loader)

    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f'  Epoch {epoch+1:>3}: train={train_loss/steps:.4f} val={val_loss:.4f} acc={val_acc:.4f}')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience = 0
        torch.save(model.state_dict(), 'data/best_tox_model.pt')
    else:
        patience += 1
        if patience >= EARLY_STOP:
            print(f'  Early stopping at epoch {epoch+1}')
            break

print(f'\nBest validation accuracy (model selection only): {best_val_acc:.4f} ({best_val_acc*100:.1f}%)')

# ─── 5. EVALUATE ON HELD-OUT TEST SET ────────────────────────
model.load_state_dict(torch.load('data/best_tox_model.pt'))
model.eval()

def _infer(loader):
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.batch)
            preds.append(out['dil'].argmax(dim=-1).cpu())
            labels.append(batch.y.view(-1).cpu())
    return torch.cat(labels).numpy(), torch.cat(preds).numpy()

from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
y_val, p_val = _infer(val_loader)
y_test, p_test = _infer(test_loader)

names = ['Ambiguous', 'No-concern', 'Less-concern', 'Most-concern']
print('\n' + '='*58)
print('DILIrank 4-class report — held-out TEST split (random, optimistic)')
print('='*58)
print(classification_report(y_test, p_test, target_names=names, zero_division=0))
print('Confusion matrix (test):')
print(confusion_matrix(y_test, p_test))

# Trivial reference model so accuracy is interpretable.
vals, cnts = np.unique(dil_y[train_idx], return_counts=True)
majority = int(vals[np.argmax(cnts)])
maj_pred = np.full_like(y_test, majority)
print(f'\nBaseline (always predict "{names[majority]}"): '
      f'acc={accuracy_score(y_test, maj_pred):.4f}  macro-F1={f1_score(y_test, maj_pred, average="macro", zero_division=0):.4f}')
print(f'GNN (held-out test):                          '
      f'acc={accuracy_score(y_test, p_test):.4f}  macro-F1={f1_score(y_test, p_test, average="macro", zero_division=0):.4f}')
print('Reminder: this random split is an OPTIMISTIC upper bound; '
      'run benchmark.py --split scaffold for the honest number.')

# ─── 6. PREDICT THE SELECTED CANDIDATE ───────────────────────
print('\n' + '='*50)
print('PREDICTING SELECTED CANDIDATE')
print('='*50)

tx_endpoints = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

# Use the same lead the rest of the pipeline selected, from one source of truth.
try:
    with open('cure_found.json') as f:
        candidate_smiles = json.load(f)['smiles']
except (OSError, KeyError, json.JSONDecodeError):
    candidate_smiles = 'CC1=CC(=O)NC2=CC=CC=C12'
print(f'  Candidate SMILES: {candidate_smiles}')
candidate_graph = mol_to_graph(candidate_smiles)
candidate_graph.batch = torch.zeros(candidate_graph.x.size(0), dtype=torch.long).to(DEVICE)

with torch.no_grad():
    out = model(candidate_graph.x.to(DEVICE), candidate_graph.edge_index.to(DEVICE),
                candidate_graph.batch)
    dil_pred = out['dil'].argmax(dim=-1).item()
    ct_pred = out['ct'].item()
    tx_probs = out['tx'].squeeze().tolist()

dil_names = ['Ambiguous', 'No-concern', 'Less-concern', 'Most-concern']
print(f'  DILI concern: {dil_names[dil_pred]} (class {dil_pred})')
print(f'  CT_TOX: {ct_pred:.3f} (>0.5=toxic)')
probs = F.softmax(out['dil'], dim=-1).squeeze().tolist()
print(f'  DILI probs: {[f"{p:.3f}" for p in probs]}')
print()
print('  Tox21 endpoint predictions:')
for name, prob in zip(tx_endpoints, tx_probs):
    flag = ' *** TOXIC' if prob > 0.5 else ''
    print(f'    {name}: {prob:.3f}{flag}')
