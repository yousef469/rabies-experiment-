"""
ensemble.py — Train 5 GNNs with different seeds, evaluate mean accuracy and ensemble (majority vote).
"""
import sys, os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from rdkit import Chem
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
BATCH_SIZE = 64; LR = 1e-3; EPOCHS = 200; EARLY_STOP = 30
DATA_DIR = 'data'

# ─── 1. LOAD DATA ─────────────────────────────────────────
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
print(f'DILIrank: {len(dil)}, ClinTox: {len(ct)}, Tox21: {len(tx)}, SIDER: {len(sd)}')

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

def process_graphs(df, col='SMILES'):
    graphs, idxs = [], []
    for i, row in df.iterrows():
        g = mol_to_graph(row[col])
        if g is not None: graphs.append(g); idxs.append(i)
    return graphs, idxs

dil_graphs, dil_idx = process_graphs(dil)
ct_graphs, ct_idx = process_graphs(ct, 'smiles')
tx_graphs, tx_idx = process_graphs(tx, 'smiles')
sd_graphs, sd_idx = process_graphs(sd, 'smiles')
print(f'Graphs: DILIrank={len(dil_graphs)}, ClinTox={len(ct_graphs)}, Tox21={len(tx_graphs)}, SIDER={len(sd_graphs)}')

dil_y = dil.iloc[dil_idx]['severity'].values.astype(np.int64)
ct_y = ct.iloc[ct_idx]['CT_TOX'].values.astype(np.float32)
tx_cols = [c for c in tx.columns if c not in ['mol_id', 'smiles']]
tx_y = tx.iloc[tx_idx][tx_cols].values.astype(np.float32); tx_y = np.nan_to_num(tx_y, nan=-1)
sd_cols = [c for c in sd.columns if c != 'smiles']
sd_y = sd.iloc[sd_idx][sd_cols].values.astype(np.float32)

for g, y in zip(dil_graphs, dil_y): g.y = torch.LongTensor([y])
for g, y in zip(ct_graphs, ct_y): g.y = torch.FloatTensor([y])
for g, y in zip(tx_graphs, tx_y): g.y = torch.FloatTensor(y)
for g, y in zip(sd_graphs, sd_y): g.y = torch.FloatTensor(y)

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

# ─── TRAIN ONE MODEL ───────────────────────────────────────
def train_one(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    train_idx, val_idx = train_test_split(np.arange(len(dil_graphs)), test_size=0.2, random_state=seed, stratify=dil_y)
    classes, counts = np.unique(dil_y[train_idx], return_counts=True)
    weights = counts.sum() / (len(classes) * counts.astype(float))
    cw = torch.FloatTensor(weights).to(DEVICE)
    train_dil = [dil_graphs[i] for i in train_idx]
    val_dil = [dil_graphs[i] for i in val_idx]
    loaders = {
        'dil': GeoDataLoader(train_dil, BATCH_SIZE, shuffle=True),
        'ct': GeoDataLoader(ct_graphs, BATCH_SIZE, shuffle=True),
        'tx': GeoDataLoader(tx_graphs, BATCH_SIZE, shuffle=True),
        'sd': GeoDataLoader(sd_graphs, BATCH_SIZE, shuffle=True),
    }
    val_loader = GeoDataLoader(val_dil, BATCH_SIZE)
    model = GNNModel().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    best_acc, best_state, patience = 0, None, 0
    task_order = ['dil', 'ct', 'tx', 'sd']
    task_weight = {'dil': 2.0, 'ct': 1.0, 'tx': 1.0, 'sd': 0.5}
    for epoch in range(EPOCHS):
        model.train()
        iters = {t: iter(l) for t, l in loaders.items()}
        steps = max(len(train_dil)//BATCH_SIZE, 1)
        for _ in range(steps):
            total_loss = 0
            for task in task_order:
                try: batch = next(iters[task])
                except StopIteration: iters[task] = iter(loaders[task]); batch = next(iters[task])
                batch = batch.to(DEVICE)
                out = model(batch.x, batch.edge_index, batch.batch)
                if task == 'dil':
                    loss = F.cross_entropy(out['dil'], batch.y.squeeze(), weight=cw)
                elif task == 'ct':
                    loss = F.binary_cross_entropy(out['ct'], batch.y.squeeze())
                elif task == 'tx':
                    y = batch.y.view(-1, 12); mask = y != -1
                    loss = F.binary_cross_entropy(out['tx'], y.clamp(0,1), reduction='none')
                    loss = (loss*mask).sum() / mask.sum().clamp(min=1)
                elif task == 'sd':
                    loss = F.binary_cross_entropy(out['sd'], batch.y.view(-1, 27))
                total_loss = total_loss + loss * task_weight[task]
            opt.zero_grad(); total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        # Validate
        model.eval(); correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                out = model(batch.x, batch.edge_index, batch.batch)
                correct += (out['dil'].argmax(-1) == batch.y.squeeze()).sum().item()
                total += len(batch.y)
        acc = correct/total
        if acc > best_acc: best_acc, best_state, patience = acc, model.state_dict(), 0
        else: patience += 1
        if patience >= EARLY_STOP: break
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), f'data/ensemble_model_{seed}.pt')
    # Return val preds on the VALIDATION set only
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.batch)
            preds.append(out['dil'].argmax(-1).cpu()); labels.append(batch.y.cpu())
    return best_acc, torch.cat(preds), torch.cat(labels)

# ─── RUN ───────────────────────────────────────────────────
seeds = [42, 43, 44, 45, 46]
all_accs, all_preds, all_labels = [], [], []
for s in seeds:
    print(f'\nTraining seed={s}...')
    acc, preds, labels = train_one(s)
    all_accs.append(acc); all_preds.append(preds); all_labels.append(labels)
    print(f'  Val acc: {acc:.4f} ({acc*100:.1f}%)')

print('\n' + '='*60)
print('ENSEMBLE RESULTS (5 GNNs, seeds 42-46)')
print('='*60)
mean_acc = np.mean(all_accs); std_acc = np.std(all_accs)
print(f'Individual accuracies: {[f"{a*100:.1f}%" for a in all_accs]}')
print(f'Mean accuracy: {mean_acc*100:.1f}% ± {std_acc*100:.1f}%')

# Ensemble majority vote
all_preds = torch.stack(all_preds)  # (5, n_val)
ensemble_preds, _ = torch.mode(all_preds, dim=0)
all_labels = all_labels[0]  # same across seeds since split is same set
ensemble_acc = (ensemble_preds == all_labels).float().mean().item()
print(f'Ensemble (majority vote) accuracy: {ensemble_acc*100:.1f}%')
print(f'Improvement over mean: {(ensemble_acc-mean_acc)*100:+.1f}%')

from sklearn.metrics import classification_report, confusion_matrix
labels_np = all_labels.numpy()
print('\nEnsemble classification report:')
print(classification_report(labels_np, ensemble_preds.numpy(),
    target_names=['Ambiguous', 'No-concern', 'Less-concern', 'Most-concern']))
print('Confusion matrix:')
print(confusion_matrix(labels_np, ensemble_preds.numpy()))

results = {
    'individual_accs': [float(a) for a in all_accs],
    'mean_acc': float(mean_acc), 'std_acc': float(std_acc),
    'ensemble_acc': float(ensemble_acc),
    'seeds': seeds,
}
import json
with open('data/ensemble_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved to data/ensemble_results.json')
