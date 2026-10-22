"""
StructurePredictor: Predict protein 3D structure from sequence using ESM-2 + TCN.
Uses ESM-2 embeddings → contact map prediction → 3D coordinate reconstruction.
Fast (seconds on CPU), no MSA needed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, json, warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device("cpu")


# ─── TCN Block (same as ProteinEncoder) ───
class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation, groups=channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation, groups=channels)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = F.gelu(self.norm1(self.conv1(x)[:, :, :x.size(2)].permute(0,2,1)))
        out = self.dropout(out).permute(0,2,1)
        out = F.gelu(self.norm2(self.conv2(out)[:, :, :x.size(2)].permute(0,2,1)))
        out = self.dropout(out).permute(0,2,1)
        return out + residual


class ContactPredictor(nn.Module):
    """Predict inter-residue contact map from ESM-2 embeddings."""

    def __init__(self, esm_dim=320, hidden=128, n_blocks=3):
        super().__init__()
        self.proj = nn.Linear(esm_dim, hidden)
        self.pairwise = nn.Linear(hidden * 2, hidden)

        self.tcn_blocks = nn.ModuleList([
            TCNBlock(hidden, kernel_size=3, dilation=2**i) for i in range(n_blocks)
        ])

        self.contact_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, esm_embeds):
        """
        esm_embeds: (batch, seq_len, esm_dim)
        Returns: (batch, seq_len, seq_len) contact probabilities
        """
        B, L, D = esm_embeds.shape
        h = self.proj(esm_embeds)

        # Pairwise features
        h_i = h.unsqueeze(2).expand(-1, -1, L, -1)
        h_j = h.unsqueeze(1).expand(-1, L, -1, -1)
        pairs = torch.cat([h_i, h_j], dim=-1)
        pair_embeds = self.pairwise(pairs)

        # Process with TCN (flatten batch+residue, treat as sequence)
        flat = pair_embeds.reshape(B * L, L, -1).permute(0, 2, 1)
        for block in self.tcn_blocks:
            flat = block(flat)
        out = flat.permute(0, 2, 1).reshape(B, L, L, -1)

        contacts = self.contact_head(out).squeeze(-1)
        return torch.sigmoid(contacts)


class CoordinateReconstructor(nn.Module):
    """Reconstruct 3D coordinates from contact map + sequence."""

    def __init__(self, n_blocks=3):
        super().__init__()
        self.contact_to_dist = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.n_iter = n_blocks

    def coords_from_dist(self, dist_matrix, seq_len):
        """MDS-like reconstruction from distance matrix."""
        # Initialize with extended chain
        coords = torch.zeros(seq_len, 3, device=dist_matrix.device)
        coords[:, 2] = torch.arange(seq_len, dtype=torch.float32, device=dist_matrix.device) * 3.8

        # Iterative optimization
        for _ in range(self.n_iter):
            pred_dist = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
            mask = (pred_dist > 0.1).float()
            error = (pred_dist - dist_matrix) * mask
            grad = error.unsqueeze(-1) * (coords.unsqueeze(0) - coords.unsqueeze(1))
            grad = grad.mean(dim=1)
            coords = coords - 0.1 * grad

        return coords

    def forward(self, contacts):
        """
        contacts: (batch, seq_len, seq_len) probabilities
        Returns: (batch, seq_len, 3) coordinates
        """
        # Convert contacts to distances
        dist_pred = self.contact_to_dist(contacts.unsqueeze(-1)).squeeze(-1)
        dist_pred = (dist_pred + dist_pred.transpose(-1, -2)) / 2

        # Scale: high contact prob = short distance
        dist_matrix = 20.0 * (1.0 - dist_pred) + 3.8
        dist_matrix = dist_matrix.clamp(min=3.8, max=60.0)

        coords_list = []
        for i in range(contacts.shape[0]):
            coords_list.append(self.coords_from_dist(dist_matrix[i], contacts.shape[1]))

        return torch.stack(coords_list)


class StructurePredictor(nn.Module):
    """Full pipeline: ESM-2 embeddings → contact map → 3D structure."""

    def __init__(self, esm_dim=320, hidden=128, contact_blocks=3, coord_blocks=3):
        super().__init__()
        self.contact_predictor = ContactPredictor(esm_dim, hidden, contact_blocks)
        self.coord_reconstructor = CoordinateReconstructor(coord_blocks)

    def forward(self, esm_embeds):
        contacts = self.contact_predictor(esm_embeds)
        coords = self.coord_reconstructor(contacts)
        return contacts, coords


def get_esm_embeddings(seq, batch_size=1):
    """Extract ESM-2 embeddings for a protein sequence."""
    from esm import pretrained

    model, alphabet = pretrained.esm2_t6_8M_UR50D()
    model = model.eval()

    batch_converter = alphabet.get_batch_converter()
    data = [("protein", seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6], return_contacts=False)
        embeddings = results["representations"][6]

    # Remove BOS/EOS tokens
    return embeddings[:, 1:-1, :]


def write_pdb(coords, seq, plddt=None, output_path="output.pdb"):
    """Write PDB file from coordinates and sequence."""
    lines = ["HEADER    PROTEIN STRUCTURE PREDICTION"]
    atom_serial = 1

    for i in range(len(seq)):
        ca = coords[i].numpy()
        if plddt is not None:
            plddt_val = plddt[i] * 100
        else:
            plddt_val = 80.0

        three = seq[i] if seq[i] in "ACDEFGHIKLMNPQRSTVWY" else "X"
        lines.append(
            f"ATOM  {atom_serial:>5}  CA  {three:>3} A{i+1:>4}"
            f"    {ca[0]:>8.3f}{ca[1]:>8.3f}{ca[2]:>8.3f}"
            f"{1.0:>6.2f}{plddt_val:>6.2f}           C  "
        )
        atom_serial += 1

    lines.append("END")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    return output_path


def predict_structure(seq, model=None):
    """Predict 3D structure of a protein sequence."""
    if model is None:
        model_path = "checkpoints_multivirus/structure_predictor.pt"
        model = StructurePredictor()
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded structure predictor (contact acc: {ckpt.get('contact_acc', '?')})")
        else:
            print("Using untrained structure predictor")

    model.eval()

    print(f"Extracting ESM-2 embeddings for {len(seq)} aa...")
    esm_embeds = get_esm_embeddings(seq)

    print("Predicting contact map + reconstructing 3D structure...")
    with torch.no_grad():
        contacts, coords = model(esm_embeds)

    contact_acc = (contacts > 0.5).float()
    return contacts, coords, model


if __name__ == "__main__":
    seq = (
        "MNIQFIINPIYAWKIMGMALILIVVLVNLVLIVTITLFVFYEGEDPTTYSYKGIFNCHTQNYRLSSTNYCPRSSSGGNR"
        "HCAYNANKHQYSVVITHYGNMLHTIKETGFNLDIITVDSSSFSNHTASFNYSVVDLIIREHYVYNAITKNLHESGLYI"
        "ATRGTYSCSVSQKLYYDYGHRSSLHLLLNDGLVEDTGLSYPLSKNYFETKITYFISAITLIIILIIILIIILKRRHRRHRRRH"
    )

    contacts, coords, model = predict_structure(seq)

    # Save
    pdb_path = write_pdb(coords[0], seq, output_path="rabies_g_predicted.pdb")
    print(f"Structure saved to {pdb_path}")

    # Save contact map
    contact_np = contacts[0].numpy()
    np.save("rabies_g_contacts.npy", contact_np)
    print(f"Contact map saved ({contact_np.shape})")

    # Pocket detection from contact map
    contact_probs = contact_np.mean(axis=0)
    low_contact = np.argsort(contact_probs)[:20]
    print(f"\nPotential binding pocket residues (lowest avg contact):")
    for idx in low_contact:
        print(f"  Pos {idx+1}: {seq[idx]} (contact={contact_probs[idx]:.3f})")
