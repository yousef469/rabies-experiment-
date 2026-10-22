"""
Benchmark: zero-shot mutation effect prediction via masked log-likelihood.
Standard ESM-1v / Tranception approach:
  1. Mask the 3 nucleotides of the mutated codon
  2. Run model → get predicted probabilities for each nucleotide
  3. Score = log P(mutant_nts) - log P(WT_nts) at masked positions
  4. Higher score = model "accepts" mutation; lower = model "rejects" it

Masked LM pretraining (which our model has) is the ideal backbone for this.
"""
import os
import csv
import math

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

from config import VOCAB_SIZE, DEVICE, MASK_TOKEN
from model import ViralTCN


GENCODE = {
    'ATA':'I','ATC':'I','ATT':'I','ATG':'M',
    'ACA':'T','ACC':'T','ACG':'T','ACT':'T',
    'AAC':'N','AAT':'N','AAA':'K','AAG':'K',
    'AGC':'S','AGT':'S','AGA':'R','AGG':'R',
    'CTA':'L','CTC':'L','CTG':'L','CTT':'L',
    'CCA':'P','CCC':'P','CCG':'P','CCT':'P',
    'CAC':'H','CAT':'H','CAA':'Q','CAG':'Q',
    'CGA':'R','CGC':'R','CGG':'R','CGT':'R',
    'GTA':'V','GTC':'V','GTG':'V','GTT':'V',
    'GCA':'A','GCC':'A','GCG':'A','GCT':'A',
    'GAC':'D','GAT':'D','GAA':'E','GAG':'E',
    'GGA':'G','GGC':'G','GGG':'G','GGT':'G',
    'TCA':'S','TCC':'S','TCG':'S','TCT':'S',
    'TTC':'F','TTT':'F','TTA':'L','TTG':'L',
    'TAC':'Y','TAT':'Y','TAA':'*','TAG':'*',
    'TGC':'C','TGT':'C','TGA':'*','TGG':'W',
}
AA_TO_CODONS = {aa: [] for aa in set(GENCODE.values())}
for codon, aa in GENCODE.items():
    AA_TO_CODONS[aa].append(codon)

NTS = ['A', 'C', 'G', 'T', 'U', 'N']
VOCAB_NT = {b: i for i, b in enumerate(NTS)}


def seq_to_tensor(seq, max_len=5000):
    seq = seq.upper().replace('-', 'N').replace('?', 'N').replace(' ', '')
    tokens = [VOCAB_NT.get(c, VOCAB_NT['N']) for c in seq]
    if len(tokens) < max_len:
        tokens = tokens + [VOCAB_NT['N']] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)


def mask_codon(seq_tensor, nuc_start):
    """Mask 3 nucleotides at nuc_start using MASK_TOKEN."""
    masked = seq_tensor.clone()
    masked[0, nuc_start:nuc_start + 3] = MASK_TOKEN
    return masked


def get_codon_for_mutation(g_cds, site, mutant_aa):
    codon_idx = site - 1
    nuc_start = codon_idx * 3
    wt_codon = g_cds[nuc_start:nuc_start + 3]
    wt_aa = GENCODE.get(wt_codon, 'X')
    if wt_aa == mutant_aa:
        return g_cds, wt_codon
    mutant_codons = AA_TO_CODONS.get(mutant_aa, [])
    if not mutant_codons:
        return g_cds, wt_codon
    best_codon = mutant_codons[0]
    best_dist = 4
    for mc in mutant_codons:
        dist = sum(1 for a, b in zip(wt_codon, mc) if a != b)
        if dist < best_dist:
            best_dist = dist
            best_codon = mc
    mutant_cds = g_cds[:nuc_start] + best_codon + g_cds[nuc_start + 3:]
    return mutant_cds, best_codon


def load_g_cds(fasta_path="data/rabies_genomes/lyssavirus_full.fasta"):
    with open(fasta_path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if 'NC_001542' in line:
            seq_lines = []
            j = i + 1
            while j < len(lines) and not lines[j].startswith('>'):
                seq_lines.append(lines[j].strip())
                j += 1
            genome = ''.join(seq_lines)
            g_region = genome[3314:4883]
            return g_region[3:]
    raise ValueError("NC_001542.1 not found in fasta")


def load_dms_data(dms_path="data/dms/all_antibodies_and_cell_entry.csv"):
    rows = []
    with open(dms_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            site = int(row['site'])
            try:
                cell_entry = float(row['cell entry']) if row['cell entry'].strip() else float('nan')
                escape = float(row['monoclonal antibodies escape']) if row['monoclonal antibodies escape'].strip() else float('nan')
            except ValueError:
                cell_entry = float('nan')
                escape = float('nan')
            natural_site = int(row['natural_sequence_site'])
            rows.append({
                'site': site,
                'wildtype': row['wildtype'],
                'mutant': row['mutant'],
                'cell_entry': cell_entry,
                'escape': escape,
                'natural_sequence_site': natural_site,
            })
    return rows


@torch.no_grad()
def compute_mutation_scores(model, g_cds, dms_rows, batch_size=32, device=DEVICE):
    """
    Masked log-likelihood scoring:
    - Mask mutated codon in G CDS
    - Run predict_masked → get logits at masked positions
    - Score = log P(mutant_nt) - log P(WT_nt) summed over 3 positions
    """
    model.eval()

    g_tensor = seq_to_tensor(g_cds).to(device)

    results = []
    batch_iter = tqdm(range(0, len(dms_rows), batch_size), desc="Scoring mutations (masked LL)")
    for i in batch_iter:
        batch = dms_rows[i:i + batch_size]

        # Create masked sequences: each has the 3 codon nts set to MASK_TOKEN
        seqs = []
        mutant_seq_strs = []  # for verification
        for row in batch:
            nat = row['natural_sequence_site']
            nuc_start = (nat - 1) * 3
            mutant_cds, mut_codon = get_codon_for_mutation(g_cds, nat, row['mutant'])
            masked = mask_codon(g_tensor, nuc_start)
            seqs.append(masked)
            mutant_seq_strs.append(mut_codon)

        stacked = torch.cat(seqs).to(device)

        # Forward pass: get logits at all positions
        logits = model.predict_masked(stacked)  # (B, L, V)

        for j, row in enumerate(batch):
            nat = row['natural_sequence_site']
            nuc_start = (nat - 1) * 3

            wt_codon = g_cds[nuc_start:nuc_start + 3]
            mut_codon_str = mutant_seq_strs[j]

            # Get logits at the 3 masked positions
            log_probs = F.log_softmax(logits[j, nuc_start:nuc_start + 3], dim=-1)  # (3, V)

            # WT log-probability
            wt_ll = 0.0
            for k, nt in enumerate(wt_codon):
                nt_idx = VOCAB_NT.get(nt, VOCAB_NT['N'])
                wt_ll += log_probs[k, nt_idx].item()

            # Mutant log-probability
            mut_ll = 0.0
            for k, nt in enumerate(mut_codon_str):
                nt_idx = VOCAB_NT.get(nt, VOCAB_NT['N'])
                mut_ll += log_probs[k, nt_idx].item()

            # Score = log P(mutant) - log P(WT)
            ll_ratio = mut_ll - wt_ll

            results.append({
                'site': row['site'],
                'wildtype': row['wildtype'],
                'mutant': row['mutant'],
                'cell_entry': row['cell_entry'],
                'escape': row['escape'],
                'll_ratio': ll_ratio,
                'wt_ll': wt_ll,
                'mut_ll': mut_ll,
            })

    return results


def compute_correlations(results):
    cell_entry = np.array([r['cell_entry'] for r in results])
    escape = np.array([r['escape'] for r in results])
    ll_ratios = np.array([r['ll_ratio'] for r in results])

    valid_cell = np.isfinite(cell_entry) & np.isfinite(ll_ratios)
    valid_esc = np.isfinite(escape) & np.isfinite(ll_ratios)

    corrs = {}
    if valid_cell.sum() > 5:
        r_cell, p_cell = spearmanr(ll_ratios[valid_cell], cell_entry[valid_cell])
        corrs['cell_entry'] = {'spearman': r_cell, 'p': p_cell, 'n': int(valid_cell.sum())}
    if valid_esc.sum() > 5:
        r_esc, p_esc = spearmanr(ll_ratios[valid_esc], escape[valid_esc])
        corrs['escape'] = {'spearman': r_esc, 'p': p_esc, 'n': int(valid_esc.sum())}
    return corrs


def main():
    print("=" * 60)
    print("DMS Benchmark: Zero-Shot Mutation Effect Prediction")
    print("=" * 60)

    print("\nLoading model...")
    model = ViralTCN(model_size="lite")
    ckpt_path = "checkpoints/best_model.pt"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    epoch = ckpt.get('epoch', '?')
    val_acc = ckpt.get('val_metrics', {}).get('acc', '?')
    print(f"Model: epoch {epoch}, val_acc {val_acc}")

    print("\nLoading G CDS...")
    g_cds = load_g_cds()
    print(f"G CDS length: {len(g_cds)} bp ({len(g_cds)//3} aa)")

    print("\nLoading DMS data...")
    dms_rows = load_dms_data()
    dms_rows = [r for r in dms_rows if r['site'] >= 1]
    print(f"Ectodomain mutations: {len(dms_rows)}")

    N = min(len(dms_rows), 2000)
    dms_sample = dms_rows[:N]
    print(f"Running on {N} mutations...")

    print("\nScoring mutations via masked log-likelihood...")
    results = compute_mutation_scores(model, g_cds, dms_sample, batch_size=32, device=DEVICE)

    print("\nComputing correlations...")
    corrs = compute_correlations(results)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    for metric, scores in corrs.items():
        print(f"  {metric}: rho={scores['spearman']:.4f} (p={scores['p']:.4f}, n={scores['n']})")

    results.sort(key=lambda r: abs(r['ll_ratio']), reverse=True)
    print(f"\n  Top 5 most surprising mutations:")
    for r in results[:5]:
        print(f"    Site {r['site']} {r['wildtype']}→{r['mutant']}: ll_ratio={r['ll_ratio']:+.4f}, cell_entry={r['cell_entry']:.3f}, escape={r['escape']:.3f}")
    print(f"\n  Bottom 5 (most accepted):")
    for r in results[-5:]:
        print(f"    Site {r['site']} {r['wildtype']}→{r['mutant']}: ll_ratio={r['ll_ratio']:+.4f}, cell_entry={r['cell_entry']:.3f}, escape={r['escape']:.3f}")

    return corrs


if __name__ == "__main__":
    main()
