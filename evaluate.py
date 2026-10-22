"""
Evaluate trained model on downstream tasks:
  1. Sequence reconstruction accuracy (masked prediction)
  2. Embedding quality (nearest neighbor, clustering)
  3. Conservation scoring (predict functional regions)
  4. Drug screening (if drug data available)
"""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from config import VOCAB_SIZE, DEVICE, MAX_SEQ_LEN
from model import ViralTCN
from dataset import create_dataloaders
from utils import set_seed


@torch.no_grad()
def evaluate_reconstruction(model, loader, device=DEVICE):
    """Evaluate masked token prediction accuracy."""
    model.eval()
    total_correct = 0
    total_masked = 0

    for masked_seq, labels in tqdm(loader, desc="Reconstruction Eval"):
        masked_seq = masked_seq.to(device)
        labels = labels.to(device)

        logits = model.predict_masked(masked_seq)
        preds = logits.argmax(dim=-1)

        mask = labels != -100
        total_correct += (preds[mask] == labels[mask]).sum().item()
        total_masked += mask.sum().item()

    acc = total_correct / max(total_masked, 1)
    print(f"Masked token prediction accuracy: {acc:.4f} ({total_correct}/{total_masked})")
    return acc


@torch.no_grad()
def compute_embeddings(model, loader, device=DEVICE):
    """Extract embeddings for all sequences in loader."""
    model.eval()
    all_embeddings = []

    for batch in tqdm(loader, desc="Extracting embeddings"):
        if isinstance(batch, (list, tuple)):
            seqs = batch[0].to(device)
        else:
            seqs = batch.to(device)

        emb = model.extract_embedding(seqs)
        all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0).numpy()


def evaluate_embedding_quality(embeddings, labels=None):
    """Evaluate embedding quality via clustering."""
    if labels is not None and len(np.unique(labels)) > 1:
        sil = silhouette_score(embeddings, labels)
        print(f"Silhouette score: {sil:.4f}")
        return {"silhouette": sil}

    kmeans = KMeans(n_clusters=min(5, len(embeddings)), random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    inertia = kmeans.inertia_
    print(f"KMeans inertia (k={len(np.unique(cluster_labels))}): {inertia:.2f}")
    return {"inertia": inertia}


@torch.no_grad()
def score_variant(model, variant_seq, conserved_regions=None, device=DEVICE):
    """Score a variant for functional impact / escape potential.

    Higher embedding distance from reference = more divergent.
    Higher PDM uncertainty = potentially more concerning.
    """
    model.eval()

    if isinstance(variant_seq, list):
        variant_seq = torch.tensor(variant_seq, dtype=torch.long)
    variant_seq = variant_seq.unsqueeze(0).to(device)

    logits, embedding = model(variant_seq)
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    confidence = probs.max(dim=-1)[0]

    scores = {
        "confidence": confidence.item(),
        "entropy": entropy.item(),
        "embedding_norm": embedding.norm(dim=-1).item(),
    }
    return scores


def screen_drugs(model, virus_seq, drug_fingerprints, drug_names=None, device=DEVICE):
    """Score multiple drugs against a virus sequence.

    Args:
        virus_seq: sequence tensor (1, L) or list
        drug_fingerprints: (N, 2048) tensor of Morgan fingerprints
        drug_names: list of N drug names (optional)

    Returns:
        sorted list of (name, score) tuples
    """
    model.eval()

    if isinstance(virus_seq, list):
        virus_seq = torch.tensor(virus_seq, dtype=torch.long)
    virus_seq = virus_seq.unsqueeze(0).to(device)
    drug_fps = torch.tensor(drug_fingerprints, dtype=torch.float32).to(device)

    with torch.no_grad():
        virus_emb = model.backbone.extract_embedding(virus_seq)
        if hasattr(model, 'drug_encoder'):
            drug_emb = model.drug_encoder(drug_fps)
            scores = model.cure_head(virus_emb.expand(len(drug_fps), -1), drug_emb)

    results = []
    for i, score in enumerate(scores.cpu().numpy()):
        name = drug_names[i] if drug_names else f"drug_{i}"
        results.append((name, float(score)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def main():
    set_seed()

    model = ViralTCN(model_size="pro")
    ckpt = torch.load("checkpoints/best_model.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(DEVICE)
    print("Loaded trained model")

    _, val_loader, _ = create_dataloaders(max_samples=50, max_seq_len=5000)
    evaluate_reconstruction(model, val_loader)

    embeddings = compute_embeddings(model, val_loader)
    evaluate_embedding_quality(embeddings)


if __name__ == "__main__":
    main()
