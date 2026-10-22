"""
pretrain_toxcast.py — Pretrain GNN backbone on ToxCast (9K compounds, 617 assays).
Saves backbone weights used by train_toxicity.py for fine-tuning.
"""
import sys, os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data as GeoData
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool
from rdkit import Chem
from sklearn.model_selection import train_test_split
import warnings, json, time
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
DATA_DIR = 'data'
BATCH_SIZE = 64
HIDDEN = 64
NUM_LAYERS = 4
EPOCHS = 30
LR = 5e-4

# ─── LOAD TOXCAST ─────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA_DIR, 'toxcast.csv'))
assay_cols = [c for c in df.columns if c != 'smiles']
print(f'ToxCast: {len(df)} compounds, {len(assay_cols)} assays')

# Build graphs
ATOMIC_NUMS = [6, 7, 8, 9, 16, 17, 35, 53, 15]

def atom_features(atom):
    atomic = [0]*len(ATOMIC_NUMS)
    a = atom.GetAtomicNum()
    if a in ATOMIC_NUMS: atomic[ATOMIC_NUMS.index(a)] = 1
    else: atomic[0] = 1
    degree = atom.GetDegree()
    deg = [0]*6
    if degree < 6: deg[degree] = 1
    chg = atom.GetFormalCharge()
    charge = [1 if chg == -1 else 0, 1 if chg == 0 else 0, 1 if chg == 1 else 0]
    num_h = atom.GetTotalNumHs()
    h = [0]*5
    if num_h < 5: h[num_h] = 1
    hyb = atom.GetHybridization()
    hyb_map = {Chem.rdchem.HybridizationType.SP: 0, Chem.rdchem.HybridizationType.SP2: 1,
               Chem.rdchem.HybridizationType.SP3: 2, Chem.rdchem.HybridizationType.SP3D: 3,
               Chem.rdchem.HybridizationType.SP3D2: 4}
    hyb_vec = [0]*5
    if hyb in hyb_map: hyb_vec[hyb_map[hyb]] = 1
    aromatic = [int(atom.GetIsAromatic())]
    return atomic + deg + charge + h + hyb_vec + aromatic

# Pre-extract labels as numpy array for speed
label_data = df[assay_cols].values.astype(np.float32)
label_data[np.isnan(label_data)] = -1.0

graphs, labels = [], []
failed = 0
t0 = time.time()
for i, (_, row) in enumerate(df.iterrows()):
    smiles = str(row['smiles']).strip()
    if smiles == 'FAIL' or smiles == 'nan' or smiles == '':
        failed += 1
        continue
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        failed += 1
        continue
    node_feats = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.FloatTensor(node_feats)
    ei = []
    for bond in mol.GetBonds():
        i_idx, j_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ei.append([i_idx, j_idx]); ei.append([j_idx, i_idx])
    if ei:
        edge_index = torch.LongTensor(ei).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    g = GeoData(x=x, edge_index=edge_index)
    g.y = torch.FloatTensor(label_data[i])
    graphs.append(g)
    labels.append(g.y)
    if (i+1) % 2000 == 0:
        print(f'  Built {i+1}/{len(df)} graphs ({time.time()-t0:.0f}s)')

print(f'Graphs: {len(graphs)}, Failed SMILES: {failed}, Time: {time.time()-t0:.0f}s')

# Split
idx = np.arange(len(graphs))
train_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=42)
train_loader = GeoDataLoader([graphs[i] for i in train_idx], BATCH_SIZE, shuffle=True)
val_loader = GeoDataLoader([graphs[i] for i in val_idx], BATCH_SIZE)

# ─── MODEL ─────────────────────────────────────────────────
class GNNBackbone(nn.Module):
    def __init__(self, node_dim=29):
        super().__init__()
        self.node_emb = nn.Linear(node_dim, HIDDEN)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(NUM_LAYERS):
            mlp = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.BatchNorm1d(HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN))
            self.convs.append(GINConv(mlp, train_eps=True))
            self.norms.append(nn.BatchNorm1d(HIDDEN))
    def forward(self, x, edge_index, batch):
        h = self.node_emb(x)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index); h = norm(h); h = F.relu(h)
        return h

class ToxCastModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = GNNBackbone()
        self.pool = lambda h, batch: torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1)
        self.fc = nn.Sequential(nn.Linear(HIDDEN*2, HIDDEN), nn.ReLU(), nn.Dropout(0.5))
        self.head = nn.Linear(HIDDEN, len(assay_cols))
    def forward(self, x, edge_index, batch):
        h = self.backbone(x, edge_index, batch)
        h = self.pool(h, batch)
        h = self.fc(h)
        return self.head(h)

model = ToxCastModel().to(DEVICE)
num_params = sum(p.numel() for p in model.parameters())
print(f'Model: {num_params:,} params ({sum(p.numel() for p in model.backbone.parameters()):,} backbone)')

# ─── TRAIN ─────────────────────────────────────────────────
def masked_bce(pred, target):
    mask = target != -1.0
    loss = F.binary_cross_entropy_with_logits(pred, target.clamp(0, 1), reduction='none')
    loss = loss * mask
    return loss.sum() / mask.sum().clamp(min=1)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
best_loss = float('inf')

t0 = time.time()
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        pred = model(batch.x, batch.edge_index, batch.batch)
        target = batch.y.view(-1, len(assay_cols)).to(DEVICE)
        loss = masked_bce(pred, target)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        train_loss += loss.item()
    scheduler.step()
    # Val
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            pred = model(batch.x, batch.edge_index, batch.batch)
            target = batch.y.view(-1, len(assay_cols)).to(DEVICE)
            loss = masked_bce(pred, target)
            val_loss += loss.item()
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f'Epoch {epoch+1:>3}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}')
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.backbone.state_dict(), os.path.join(DATA_DIR, 'pretrained_backbone.pt'))

elapsed = time.time() - t0
print(f'\nDone in {elapsed:.0f}s. Best val_loss={best_loss:.4f}')
print(f'Backbone saved to {DATA_DIR}/pretrained_backbone.pt')
