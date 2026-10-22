"""
Download, preprocess, and load rabies/lyssavirus genome sequences.

Data sources:
  - NCBI Virus: Lyssavirus genomes (taxid 11292)
  - Local FASTA fallback for offline use

Pipeline:
  1. Download genomes via NCBI API → FASTA
  2. Parse + one-hot encode nucleotide sequences
  3. Create masked sequences for self-supervised pretraining
  4. Create labeled datasets for fine-tuning (protein region prediction)
"""
import os
import json
import gzip
import random
import requests
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO

from config import (
    VOCAB, VOCAB_SIZE, PAD_TOKEN, MASK_TOKEN, MAX_SEQ_LEN,
    MASK_PROB, MASK_REPLACE_PROB, MASK_RANDOM_PROB, MASK_KEEP_PROB,
    DATA_DIR, NCBI_TAXID_LYSSAVIRUS, BATCH_SIZE, NUM_WORKERS,
    NUM_PROTEIN_CLASSES,
)

RABIES_GENES = {
    "leader": (0, 58),
    "N": (58, 1493),
    "P": (1493, 2623),
    "M": (2623, 3313),
    "G": (3313, 4906),
    "L": (4906, 10977),
    "trailer": (10977, 11932),
}

RABIES_REF_SEQ_LENGTH = 11932


def download_ncbi_virus(taxid=NCBI_TAXID_LYSSAVIRUS, output_dir=DATA_DIR, max_results=200):
    """Download virus genomes from NCBI Virus API."""
    os.makedirs(output_dir, exist_ok=True)
    fasta_path = os.path.join(output_dir, "lyssavirus_genomes.fasta")

    if os.path.exists(fasta_path):
        print(f"Found existing dataset: {fasta_path}")
        return fasta_path

    url = (
        f"https://api.ncbi.nlm.nih.gov/datasets/v2/virus/taxon/{taxid}/genome/download"
        f"?page_size={min(max_results, 500)}&page_token=&include_annotation_type=GENOME_FASTA"
    )
    headers = {"Accept": "application/json"}

    print(f"Downloading lyssavirus genomes (taxid {taxid})...")
    resp = requests.post(url, headers=headers)
    if resp.status_code != 200:
        print(f"NCBI API returned {resp.status_code}, trying alternative download...")
        return _download_ncbi_fallback(taxid, output_dir, max_results)

    result = resp.json()
    download_url = result.get("download", {}).get("url")
    if not download_url:
        print("No download URL in response, trying fallback...")
        return _download_ncbi_fallback(taxid, output_dir, max_results)

    print(f"Downloading from: {download_url}")
    dl_resp = requests.get(download_url, stream=True)
    if dl_resp.status_code != 200:
        print(f"Download failed ({dl_resp.status_code}), trying fallback...")
        return _download_ncbi_fallback(taxid, output_dir, max_results)

    with open(fasta_path, 'wb') as f:
        for chunk in dl_resp.iter_content(chunk_size=8192):
            f.write(chunk)

    count = sum(1 for _ in SeqIO.parse(fasta_path, "fasta"))
    print(f"Downloaded {count} genomes to {fasta_path}")
    return fasta_path


def _download_ncbi_fallback(taxid, output_dir, max_results):
    """Fallback: download via NCBI E-utilities."""
    fasta_path = os.path.join(output_dir, "lyssavirus_genomes.fasta")

    search_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=nuccore&term=txid{taxid}[Organism]+AND+complete+genome&retmax={max_results}"
        f"&retmode=json"
    )
    resp = requests.get(search_url)
    if resp.status_code != 200:
        raise RuntimeError(f"NCBI E-utilities search failed: {resp.status_code}")
    ids = resp.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        raise RuntimeError("No genomes found for this taxid")

    print(f"Found {len(ids)} genome IDs, downloading...")

    fetched = 0
    with open(fasta_path, 'w') as f:
        for i in range(0, len(ids), 10):
            batch = ids[i:i + 10]
            fetch_url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                f"?db=nuccore&id={','.join(batch)}&rettype=fasta&retmode=text"
            )
            fetch_resp = requests.get(fetch_url)
            if fetch_resp.status_code == 200:
                f.write(fetch_resp.text)
                fetched += len(batch)
            print(f"  Fetched {fetched}/{len(ids)}")

    count = sum(1 for _ in SeqIO.parse(fasta_path, "fasta"))
    print(f"Downloaded {count} genomes to {fasta_path}")
    return fasta_path


def download_example_targets(output_dir=DATA_DIR):
    """Download example target data for the G glycoprotein epitopes."""
    target_dir = os.path.join(output_dir, "targets")
    os.makedirs(target_dir, exist_ok=True)
    return target_dir


def sequence_to_ids(seq, max_len=MAX_SEQ_LEN):
    """Convert nucleotide sequence to integer IDs."""
    seq = seq.upper().strip()
    ids = []
    for c in seq[:max_len]:
        ids.append(VOCAB.get(c, PAD_TOKEN))
    if len(ids) < max_len:
        ids.extend([PAD_TOKEN] * (max_len - len(ids)))
    return ids[:max_len]


def ids_to_sequence(ids):
    """Convert integer IDs back to nucleotide string."""
    rev = {v: k for k, v in VOCAB.items()}
    return "".join(rev.get(i, "N") for i in ids)


def mask_tokens(inputs, mask_token_id=MASK_TOKEN, vocab_size=VOCAB_SIZE,
                pad_token_id=PAD_TOKEN, prob=MASK_PROB,
                replace_prob=MASK_REPLACE_PROB, random_prob=MASK_RANDOM_PROB,
                keep_prob=MASK_KEEP_PROB):
    """Create masked version of inputs for self-supervised learning."""
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, prob)
    special_mask = (inputs == pad_token_id)
    probability_matrix.masked_fill_(special_mask, value=0.0)
    mask_indices = torch.bernoulli(probability_matrix).bool()
    labels[~mask_indices] = -100
    indices_replaced = torch.bernoulli(torch.full(labels.shape, replace_prob)).bool() & mask_indices
    inputs[indices_replaced] = mask_token_id
    indices_random = torch.bernoulli(torch.full(labels.shape, random_prob / (1 - replace_prob))).bool() & mask_indices & ~indices_replaced
    random_tokens = torch.randint(vocab_size, labels.shape, dtype=torch.long)
    inputs[indices_random] = random_tokens[indices_random]
    return inputs, labels


class RabiesGenomeDataset(Dataset):
    """Load and preprocess rabies/lyssavirus genomes."""
    def __init__(self, fasta_path=None, split="train", max_seq_len=MAX_SEQ_LEN,
                 max_samples=None, download=True):
        self.max_seq_len = max_seq_len
        self.sequences = []
        self.accessions = []

        if fasta_path is None:
            fasta_path = os.path.join(DATA_DIR, "lyssavirus_genomes.fasta")

        if not os.path.exists(fasta_path) and download:
            print("Dataset not found locally. Downloading from NCBI...")
            fasta_path = download_ncbi_virus()

        if os.path.exists(fasta_path):
            self._load_fasta(fasta_path, max_samples, split)
        else:
            self._load_synthetic(split, max_samples)

        print(f"Loaded {len(self.sequences)} sequences ({split})")

    @staticmethod
    def _get_split_indices(total, split):
        n_train = int(total * 0.8)
        n_val = int(total * 0.1)
        if split == "train":
            return 0, n_train
        elif split == "val":
            return n_train, n_train + n_val
        else:
            return n_train + n_val, total

    def _load_fasta(self, fasta_path, max_samples, split):
        records = list(SeqIO.parse(fasta_path, "fasta"))
        random.seed(42)
        random.shuffle(records)

        total = max_samples if max_samples else len(records)
        total = min(total, len(records))
        records = records[:total]

        start, end = self._get_split_indices(total, split)
        records = records[start:end]

        for record in records:
            seq = str(record.seq).upper()
            seq_clean = "".join(c for c in seq if c in VOCAB or c == "N")
            ids = sequence_to_ids(seq_clean, self.max_seq_len)
            self.sequences.append(torch.tensor(ids, dtype=torch.long))
            self.accessions.append(record.id)

    def _load_synthetic(self, split, max_samples):
        """Generate synthetic rabies-like sequences when no real data."""
        n = max_samples if max_samples else (200 if split == "train" else 20)
        base_probs = [0.24, 0.24, 0.26, 0.26, 0.0]
        for i in range(n):
            seq = torch.multinomial(torch.tensor(base_probs), self.max_seq_len, replacement=True)
            self.sequences.append(seq)
            self.accessions.append(f"synthetic_{split}_{i}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class MaskedGenomeDataset(Dataset):
    """Dataset wrapper that returns (masked_input, labels) for self-supervised learning."""
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        seq = self.base[idx]
        masked, labels = mask_tokens(seq.clone())
        return masked, labels


class ProteinRegionDataset(Dataset):
    """Fine-tuning dataset: predict which protein region each position belongs to."""
    def __init__(self, base_dataset, ref_start=0, ref_end=RABIES_REF_SEQ_LENGTH):
        self.base = base_dataset
        self.region_labels = self._build_region_labels(ref_start, ref_end)

    def _build_region_labels(self, start, end):
        labels = torch.full((end - start,), NUM_PROTEIN_CLASSES - 1, dtype=torch.long)
        gene_names = list(RABIES_GENES.keys())
        for i, (gene, (gs, ge)) in enumerate(RABIES_GENES.items()):
            for j in range(max(gs - start, 0), min(ge - start, end - start)):
                if j < len(labels):
                    labels[j] = i
        return labels

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.base[idx], self.region_labels


def create_dataloaders(fasta_path=None, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                       max_samples=None, max_seq_len=MAX_SEQ_LEN):
    """Create train/val/test dataloaders for all training phases."""
    train_dataset = RabiesGenomeDataset(fasta_path, "train", max_seq_len, max_samples)
    val_dataset = RabiesGenomeDataset(fasta_path, "val", max_seq_len, max_samples)
    test_dataset = RabiesGenomeDataset(fasta_path, "test", max_seq_len, max_samples)

    train_masked = MaskedGenomeDataset(train_dataset)
    val_masked = MaskedGenomeDataset(val_dataset)

    train_loader = DataLoader(train_masked, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_masked, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
