"""
Phase 2 fine-tuning: predict protein region labels per position.
Uses the pretrained backbone and adds a position-wise classifier head.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from config import (
    DEVICE, VOCAB_SIZE, NUM_PROTEIN_CLASSES, LR, WEIGHT_DECAY, EPOCHS,
    RABIES_GENES,
)
from model import ViralTCN
from dataset import RabiesGenomeDataset
from utils import set_seed, count_parameters, Checkpointer


class FineTuneModel(nn.Module):
    """Pretrained TCN + position-wise protein classifier head."""
    def __init__(self, backbone, num_classes=NUM_PROTEIN_CLASSES):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Conv1d(backbone.embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        B, L = x.shape
        x = self.backbone.embedding(x)
        x = x.permute(0, 2, 1)
        x = x + self.backbone.pos_embed
        for block in self.backbone.blocks:
            x = block(x)
        logits = self.classifier(x)
        return logits.permute(0, 2, 1)


def create_region_labels(seq_len, ref_seq_length=11932):
    """Create position-wise region labels based on rabies reference genome."""
    labels = torch.full((seq_len,), NUM_PROTEIN_CLASSES - 1, dtype=torch.long)
    gene_names = list(RABIES_GENES.keys())
    for i, (name, (start, end)) in enumerate(RABIES_GENES.items()):
        scale = seq_len / ref_seq_length
        s = int(start * scale)
        e = int(end * scale)
        for j in range(max(0, s), min(e, seq_len)):
            labels[j] = i
    return labels


@torch.no_grad()
def generate_region_labels_for_dataset(model, loader, device=DEVICE):
    """Generate pseudo-labels using the pretrained PDM."""
    model.eval()
    all_preds = []
    for batch in tqdm(loader, desc="Generating region labels"):
        if isinstance(batch, (list, tuple)):
            seqs = batch[0].to(device)
        else:
            seqs = batch.to(device)
        logits, _ = model(seqs)
        preds = logits.argmax(dim=-1)
        all_preds.append(preds.cpu())
    return torch.cat(all_preds)


def main():
    set_seed()
    print(f"Device: {DEVICE}")

    backbone = ViralTCN(model_size="pro")
    ckpt_path = "checkpoints/best_model.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        backbone.load_state_dict(ckpt['model_state_dict'])
        print("Loaded pretrained backbone")
    else:
        print("No pretrained checkpoint found, using random init backbone")

    model = FineTuneModel(backbone).to(DEVICE)
    trainable, total = count_parameters(model)
    print(f"Fine-tune params: {trainable:,} trainable / {total:,} total")

    dataset = RabiesGenomeDataset(split="train", max_samples=100, max_seq_len=5000)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    region_labels = create_region_labels(5000).to(DEVICE)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        total_loss = 0.0
        all_preds, all_true = [], []

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/10"):
            seqs = batch.to(DEVICE)
            B, L = seqs.shape
            logits = model(seqs)

            labels = region_labels.unsqueeze(0).expand(B, -1)
            loss = criterion(logits.reshape(-1, NUM_PROTEIN_CLASSES), labels.reshape(-1))
            orth_loss = model.backbone.pdm.orthogonality_loss()
            total = loss + 0.05 * orth_loss

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * B
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_true.extend(labels.cpu().numpy().flatten())

        avg_loss = total_loss / len(loader.dataset)
        acc = accuracy_score(all_true, all_preds)
        print(f"  Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

    torch.save(model.state_dict(), "checkpoints/finetuned_region.pt")
    print("Fine-tuned model saved")


if __name__ == "__main__":
    main()
