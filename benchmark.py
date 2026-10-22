"""
benchmark.py — Head-to-head toxicity benchmark: GNN vs fingerprint DNN (Xu 2020 style).

Protocol
--------
Both models are trained and evaluated on the SAME DILIrank molecules with the SAME
3-way split (train / val / test). The model is selected on the validation split and
the reported number is measured on the held-out test split — never on the split used
for model selection.

Two evaluation regimes are supported because they answer different questions:

  --split random    Stratified random split. Optimistic: structurally similar molecules
                    can appear in both train and test, so scores are an upper bound.
  --split scaffold  Murcko-scaffold split. Realistic: every scaffold in test is unseen
                    during training. This is the honest "would this work on a new drug?"
                    estimate and is the number that should be quoted.

Two trivial reference models are reported next to each result so the accuracy is
interpretable:
  * majority   — always predicts the most frequent training class
  * stratified — samples a class from the training class distribution (seeded)

Usage
-----
    python benchmark.py --split scaffold
    python benchmark.py --split random
"""
import argparse
import json
import os
import random
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

DATA_DIR = 'data'
DILI_XLSX = os.path.join(DATA_DIR, 'Drug Induced Liver Injury Rank (DILIrank 2.0) Dataset  FDA.xlsx')
SEV_MAP = {'vMost-DILI-concern': 3, 'vMOST-DILI-concern': 3, 'vLess-DILI-concern': 2,
           'vNo-DILI-concern': 1, 'vNo-DILI-Concern': 1, 'Ambiguous-DILI-concern': 0}
CLASS_NAMES = ['Ambiguous', 'No-concern', 'Less-concern', 'Most-concern']


# ─── DATA ─────────────────────────────────────────────────────
def load_dilirank():
    """Return (smiles: list[str], labels: np.ndarray[int64]). One row per molecule."""
    dil = pd.read_excel(DILI_XLSX, sheet_name='version 2', skiprows=2)
    dil.columns = ['LTKBID', 'CompoundName', 'SeverityClass', 'LabelSection',
                   'vDILI-Concern', 'Comment']
    smi_df = pd.read_csv(os.path.join(DATA_DIR, 'dilirank_smiles.csv'))
    smi_df = smi_df[smi_df['Status'] == 'OK']
    dil = dil.merge(smi_df, left_on='CompoundName', right_on='DrugName', how='inner')
    dil['severity'] = dil['vDILI-Concern'].map(SEV_MAP)
    dil = dil[dil['severity'].notna()]
    return dil['SMILES'].tolist(), dil['severity'].values.astype(np.int64)


# ─── SPLITS ───────────────────────────────────────────────────
def scaffold_of(smiles):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ''
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ''


def random_split(labels, frac=(0.6, 0.2, 0.2), seed=42):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(labels))
    train, rest = train_test_split(idx, test_size=frac[1] + frac[2], random_state=seed,
                                   stratify=labels)
    val, test = train_test_split(rest, test_size=frac[2] / (frac[1] + frac[2]),
                                 random_state=seed, stratify=labels[rest])
    return list(train), list(val), list(test)


def scaffold_split(smiles, frac=(0.6, 0.2, 0.2), seed=42):
    """Group molecules by Murcko scaffold; whole scaffolds go to a single split.

    Scaffold groups are visited largest-first and each is assigned to the split with
    the largest remaining deficit, so the three splits hit the requested fractions
    while no scaffold is shared between splits.
    """
    groups = {}
    for i, s in enumerate(smiles):
        groups.setdefault(scaffold_of(s), []).append(i)
    # Deterministic order: large scaffolds first, ties broken by a seeded shuffle so
    # the split is reproducible but not biased toward alphabetically early scaffolds.
    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda kv: -len(kv[1]))

    n = len(smiles)
    targets = [frac[0] * n, frac[1] * n, frac[2] * n]
    splits = [[], [], []]
    counts = [0, 0, 0]
    for _, idxs in items:
        deficit = [targets[k] - counts[k] for k in range(3)]
        k = int(np.argmax(deficit))
        splits[k].extend(idxs)
        counts[k] += len(idxs)
    return splits[0], splits[1], splits[2]


# ─── FEATURIZATION ────────────────────────────────────────────
ATOMIC_NUMS = [6, 7, 8, 9, 16, 17, 35, 53, 15]


def atom_features(atom):
    from rdkit import Chem
    atomic = [0] * len(ATOMIC_NUMS)
    if atom.GetAtomicNum() in ATOMIC_NUMS:
        atomic[ATOMIC_NUMS.index(atom.GetAtomicNum())] = 1
    else:
        atomic[0] = 1
    degree = atom.GetDegree()
    deg = [0] * 6
    if degree < 6:
        deg[degree] = 1
    chg = atom.GetFormalCharge()
    charge = [1 if chg == -1 else 0, 1 if chg == 0 else 0, 1 if chg == 1 else 0]
    num_h = atom.GetTotalNumHs()
    h = [0] * 5
    if num_h < 5:
        h[num_h] = 1
    hyb = atom.GetHybridization()
    hyb_map = {Chem.rdchem.HybridizationType.SP: 0, Chem.rdchem.HybridizationType.SP2: 1,
               Chem.rdchem.HybridizationType.SP3: 2, Chem.rdchem.HybridizationType.SP3D: 3,
               Chem.rdchem.HybridizationType.SP3D2: 4}
    hyb_vec = [0] * 5
    if hyb in hyb_map:
        hyb_vec[hyb_map[hyb]] = 1
    aromatic = [int(atom.GetIsAromatic())]
    return atomic + deg + charge + h + hyb_vec + aromatic


def build_features(smiles_list, labels):
    """Return (graphs, fingerprints, labels) skipping unparseable SMILES."""
    import torch
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from torch_geometric.data import Data as GeoData

    graphs, fps, ys, kept_smiles = [], [], [], []
    for smi, y in zip(smiles_list, labels):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024), dtype=np.float32)
        desc = np.array([Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
                         Descriptors.TPSA(mol), Descriptors.NumHDonors(mol),
                         Descriptors.NumHAcceptors(mol), Descriptors.NumRotatableBonds(mol),
                         Descriptors.NumAromaticRings(mol), Descriptors.FractionCSP3(mol),
                         Descriptors.NHOHCount(mol)], dtype=np.float32)
        x = torch.FloatTensor([atom_features(a) for a in mol.GetAtoms()])
        ei = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            ei += [[i, j], [j, i]]
        edge_index = (torch.LongTensor(ei).t().contiguous() if ei
                      else torch.zeros((2, 0), dtype=torch.long))
        g = GeoData(x=x, edge_index=edge_index)
        g.y = torch.LongTensor([int(y)])
        graphs.append(g)
        fps.append(np.concatenate([fp, desc]))
        ys.append(int(y))
        kept_smiles.append(smi)
    return graphs, np.array(fps), np.array(ys), kept_smiles


# ─── MODELS ───────────────────────────────────────────────────
def make_models(node_dim=29):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool

    class FingerprintDNN(nn.Module):
        def __init__(self, input_dim=1033, hidden=512):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(hidden // 2, hidden // 8), nn.BatchNorm1d(hidden // 8), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(hidden // 8, 4),
            )

        def forward(self, x):
            return self.net(x)

    class GNNModel(nn.Module):
        def __init__(self, node_dim=29, hidden=64, num_layers=4):
            super().__init__()
            self.node_emb = nn.Linear(node_dim, hidden)
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            for _ in range(num_layers):
                mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden),
                                    nn.ReLU(), nn.Linear(hidden, hidden))
                self.convs.append(GINConv(mlp, train_eps=True))
                self.norms.append(nn.BatchNorm1d(hidden))
            self.fc = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(0.5))
            self.out = nn.Linear(hidden, 4)

        def forward(self, x, edge_index, batch):
            h = self.node_emb(x)
            for conv, norm in zip(self.convs, self.norms):
                h = norm(conv(h, edge_index))
                h = F.relu(h)
            h_pool = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1)
            return self.out(self.fc(h_pool))

    return FingerprintDNN, GNNModel


# ─── METRICS ──────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob=None):
    from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0)
    out = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'per_class': {CLASS_NAMES[i]: {'precision': float(prec[i]), 'recall': float(rec[i]),
                                       'f1': float(f1[i]), 'support': int(sup[i])}
                      for i in range(4)},
        'auroc_macro': None,
    }
    if y_prob is not None:
        from sklearn.metrics import roc_auc_score
        try:
            out['auroc_macro'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr',
                                                     average='macro', labels=[0, 1, 2, 3]))
        except ValueError:
            out['auroc_macro'] = None
    return out


def baselines(y_train, y_test, seed=42):
    """Majority-class and stratified-random reference scores on the test set."""
    vals, counts = np.unique(y_train, return_counts=True)
    majority = vals[int(np.argmax(counts))]
    maj_pred = np.full_like(y_test, majority)
    rng = np.random.RandomState(seed)
    priors = counts / counts.sum()
    rand_pred = rng.choice(vals, size=len(y_test), p=priors)
    return {'majority_class': compute_metrics(y_test, maj_pred),
            'stratified_random': compute_metrics(y_test, rand_pred)}


# ─── TRAINING ─────────────────────────────────────────────────
def train_and_eval(split_name, epochs=200, seed=42, frac=(0.6, 0.2, 0.2)):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch_geometric.loader import DataLoader as GeoDataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    smiles, labels = load_dilirank()
    if split_name == 'scaffold':
        tr, va, te = scaffold_split(smiles, frac, seed)
    else:
        tr, va, te = random_split(labels, frac, seed)

    graphs, fps, ys, smiles = build_features(smiles, labels)
    # Rebuild index lists against the kept molecules: build_features preserves order,
    # so positional indices remain valid as long as nothing was dropped.
    assert len(smiles) == len(labels), 'SMILES dropped during featurization; rebuild indices.'

    train_g = [graphs[i] for i in tr]
    val_g = [graphs[i] for i in va]
    test_g = [graphs[i] for i in te]
    train_y, val_y, test_y = ys[tr], ys[va], ys[te]

    # Inverse-frequency class weights, padded to all 4 classes (a class may be absent
    # from the training split of a scaffold split).
    classes, counts = np.unique(train_y, return_counts=True)
    weights = counts.sum() / (len(classes) * counts.astype(float))
    cw = torch.ones(4)
    for c, w in zip(classes, weights):
        cw[c] = w
    cw = cw.to(device)

    g_train = GeoDataLoader(train_g, 64, shuffle=True)
    g_val = GeoDataLoader(val_g, 64)
    g_test = GeoDataLoader(test_g, 64)

    class SimpleDS(Dataset):
        def __init__(self, x, y):
            self.x = torch.FloatTensor(x)
            self.y = torch.LongTensor(y)

        def __len__(self):
            return len(self.x)

        def __getitem__(self, i):
            return self.x[i], self.y[i]

    d_train = DataLoader(SimpleDS(fps[tr], train_y), 64, shuffle=True)
    d_val = DataLoader(SimpleDS(fps[va], val_y), 64)
    d_test = DataLoader(SimpleDS(fps[te], test_y), 64)

    FingerprintDNN, GNNModel = make_models()

    def fit(model, train_loader, val_loader, is_gnn, w):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        best, best_state, patience, t0 = 0.0, None, 0, time.time()
        for ep in range(epochs):
            model.train()
            for batch in train_loader:
                opt.zero_grad()
                if is_gnn:
                    batch = batch.to(device)
                    logits = model(batch.x, batch.edge_index, batch.batch)
                    y = batch.y.view(-1)
                else:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                F.cross_entropy(logits, y, weight=w).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for batch in val_loader:
                    if is_gnn:
                        batch = batch.to(device)
                        pred = model(batch.x, batch.edge_index, batch.batch).argmax(-1)
                        y = batch.y.view(-1)
                    else:
                        x, y = batch
                        pred = model(x.to(device)).argmax(-1)
                        y = y.to(device)
                    correct += (pred == y).sum().item()
                    total += len(y)
            acc = correct / max(total, 1)
            if acc > best:
                best, best_state, patience = acc, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                patience += 1
            if patience >= 40:
                break
        model.load_state_dict(best_state)
        return best, time.time() - t0

    def infer(model, loader, is_gnn):
        preds, probs, truth = [], [], []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                if is_gnn:
                    batch = batch.to(device)
                    logits = model(batch.x, batch.edge_index, batch.batch)
                    y = batch.y.view(-1)
                else:
                    x, y = batch
                    logits = model(x.to(device))
                    y = y.to(device)
                preds.append(logits.argmax(-1).cpu())
                probs.append(F.softmax(logits, dim=-1).cpu())
                truth.append(y.cpu())
        return (torch.cat(truth).numpy(), torch.cat(preds).numpy(),
                torch.cat(probs).numpy())

    dnn = FingerprintDNN().to(device)
    dnn_val, dnn_t = fit(dnn, d_train, d_val, False, cw)
    gnn = GNNModel().to(device)
    gnn_val, gnn_t = fit(gnn, g_train, g_val, True, cw)

    y_true, dnn_pred, dnn_prob = infer(dnn, d_test, False)
    _, gnn_pred, gnn_prob = infer(gnn, g_test, True)

    return {
        'split': split_name,
        'n_total': len(ys),
        'n_train': len(tr), 'n_val': len(va), 'n_test': len(te),
        'n_train_scaffolds': len(set(scaffold_of(s) for s in smiles)) if split_name == 'scaffold' else None,
        'val_accuracy': {'DNN': dnn_val, 'GNN': gnn_val},
        'test': {'DNN': compute_metrics(y_true, dnn_pred, dnn_prob),
                 'GNN': compute_metrics(y_true, gnn_pred, gnn_prob)},
        'baselines': baselines(train_y, test_y, seed),
        'params': {'DNN': sum(p.numel() for p in dnn.parameters()),
                   'GNN': sum(p.numel() for p in gnn.parameters())},
        'train_seconds': {'DNN': dnn_t, 'GNN': gnn_t},
    }


def print_report(res):
    b = res['baselines']
    print('\n' + '=' * 72)
    print(f"TOXICITY BENCHMARK — split={res['split']}  "
          f"(train {res['n_train']} / val {res['n_val']} / test {res['n_test']})")
    print('=' * 72)
    print(f"{'Model':<24}{'Accuracy':>10}{'Macro-F1':>10}{'Wtd-F1':>10}{'AUROC':>9}{'Params':>10}")
    print('-' * 72)
    rows = [('Majority class', b['majority_class'], None),
            ('Stratified random', b['stratified_random'], None),
            ('Fingerprint DNN', res['test']['DNN'], res['params']['DNN']),
            ('GNN (ours)', res['test']['GNN'], res['params']['GNN'])]
    for name, m, params in rows:
        auroc = m['auroc_macro']
        print(f"{name:<24}{m['accuracy']:>10.4f}{m['f1_macro']:>10.4f}"
              f"{m['f1_weighted']:>10.4f}{(f'{auroc:.4f}' if auroc else '   n/a'):>9}"
              f"{(f'{params:,}' if params else ''):>10}")
    print('-' * 72)
    for name in ('DNN', 'GNN'):
        pc = res['test'][name]['per_class']
        print(f"  {name} per-class F1: " + '  '.join(
            f"{k}={v['f1']:.2f}(n={v['support']})" for k, v in pc.items()))
    print('=' * 72)
    print('Reminder: "scaffold" is the honest estimate (unseen scaffolds in test); '
          '"random" is optimistic.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['random', 'scaffold', 'both'], default='scaffold')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    splits = ['random', 'scaffold'] if args.split == 'both' else [args.split]
    for sp in splits:
        res = train_and_eval(sp, epochs=args.epochs, seed=args.seed)
        print_report(res)
        out_path = os.path.join(DATA_DIR, f'benchmark_results_{sp}.json')
        with open(out_path, 'w') as f:
            json.dump(res, f, indent=2)
        print(f'Saved -> {out_path}')
