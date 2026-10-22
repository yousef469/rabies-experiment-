"""
1D TCN + DepthwiseSeparableConv1D + PDM for viral sequence understanding.
Adapted from top-3 cancer detection architecture (2D → 1D).

Two configs:
  Lite (~3M)  — trains on CPU/MacBook
  Pro  (~14M) — needs GPU, matches cancer model design philosophy

Training phases:
  Phase 1: Self-supervised masked prediction (learn viral grammar)
  Phase 2: Fine-tune on downstream tasks (protein prediction, conservation scoring)
  Phase 3: Drug screening head (given variant + drug → efficacy)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    VOCAB_SIZE, KERNEL_SIZE, PDM_TEMPERATURE, NUM_PROTEIN_CLASSES,
)


class DepthwiseSeparableConv1DBlock(nn.Module):
    def __init__(self, dim, kernel_size=9, dilation=1, expand_ratio=2, dropout=0.1):
        super().__init__()
        hidden_dim = dim * expand_ratio
        padding = dilation * (kernel_size - 1) // 2

        self.depthwise = nn.Conv1d(
            dim, dim, kernel_size=kernel_size,
            padding=padding, dilation=dilation, groups=dim, bias=False,
        )
        self.expand = nn.Conv1d(dim, hidden_dim, kernel_size=1, bias=False)
        self.project = nn.Conv1d(hidden_dim, dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.depthwise(x)
        x = self.expand(x)
        x = F.gelu(x)
        x = self.project(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        return x + residual


class GlobalAttention1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        B, C, L = x.shape
        x_perm = x.permute(0, 2, 1)
        q = self.query(x_perm)
        k = self.key(x_perm)
        v = self.value(x_perm)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return out.mean(dim=1)


class PrototypeDistributionModule(nn.Module):
    """Multi-subspace prototype learning — exact replica from cancer model adapted for 1D."""
    def __init__(self, input_dim, num_subspaces=4, subspace_dim=32,
                 num_classes=NUM_PROTEIN_CLASSES, temperature=PDM_TEMPERATURE):
        super().__init__()
        self.num_subspaces = num_subspaces
        self.subspace_dim = subspace_dim
        proto_dim = num_subspaces * subspace_dim

        self.subspace_proj = nn.ModuleList([
            nn.Linear(input_dim, subspace_dim, bias=False)
            for _ in range(num_subspaces)
        ])
        self.prototypes = nn.Parameter(torch.randn(num_classes, proto_dim))
        nn.init.normal_(self.prototypes, std=0.01)
        self.prototypes.data = F.normalize(self.prototypes.data, dim=-1)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))

    def forward(self, x):
        subspaces = [proj(x) for proj in self.subspace_proj]
        d = F.normalize(torch.cat(subspaces, dim=-1), dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        logits = torch.mm(d, p.t())
        temp = self.log_temperature.exp().clamp(min=0.01, max=1.0)
        return logits / temp, d

    def orthogonality_loss(self):
        loss = 0.0
        count = 0
        for i in range(self.num_subspaces):
            for j in range(i + 1, self.num_subspaces):
                Wi = F.normalize(self.subspace_proj[i].weight, dim=-1)
                Wj = F.normalize(self.subspace_proj[j].weight, dim=-1)
                loss += (Wi @ Wj.t()).norm(p='fro') ** 2
                count += 1
        return loss / count if count > 0 else 0.0


CONFIG_LITE = {
    "embed_dim": 256, "num_blocks": 6, "expand_ratio": 2,
    "dropout": 0.1, "num_subspaces": 4, "subspace_dim": 32,
}
CONFIG_PRO = {
    "embed_dim": 512, "num_blocks": 10, "expand_ratio": 3,
    "dropout": 0.15, "num_subspaces": 8, "subspace_dim": 48,
}


class ViralTCN(nn.Module):
    def __init__(self, model_size="pro"):
        super().__init__()
        self.model_size = model_size
        cfg = CONFIG_PRO if model_size == "pro" else CONFIG_LITE

        self.embed_dim = cfg["embed_dim"]
        self.embedding = nn.Embedding(VOCAB_SIZE + 1, cfg["embed_dim"], padding_idx=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg["embed_dim"], 1))

        self.blocks = nn.ModuleList([
            DepthwiseSeparableConv1DBlock(
                dim=cfg["embed_dim"],
                kernel_size=KERNEL_SIZE,
                dilation=2 ** i,
                expand_ratio=cfg["expand_ratio"],
                dropout=cfg["dropout"],
            )
            for i in range(cfg["num_blocks"])
        ])

        self.global_attn = GlobalAttention1D(cfg["embed_dim"])

        self.pdm = PrototypeDistributionModule(
            input_dim=cfg["embed_dim"],
            num_subspaces=cfg["num_subspaces"],
            subspace_dim=cfg["subspace_dim"],
            num_classes=NUM_PROTEIN_CLASSES,
        )

        self.lm_head = nn.Linear(cfg["embed_dim"], VOCAB_SIZE)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2 and 'norm' not in name and 'subspace_proj' not in name:
                if 'lm_head' in name or 'embedding' in name:
                    nn.init.normal_(param, mean=0.0, std=0.02)
                elif 'depthwise' in name and 'weight' in name:
                    nn.init.normal_(param, mean=0.0, std=0.01)
                elif 'expand' in name or 'project' in name:
                    nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        B, L = x.shape
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        pooled = self.global_attn(x)
        logits, embedding = self.pdm(pooled)
        return logits, embedding

    def forward_with_repr(self, x):
        B, L = x.shape
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return x

    def predict_masked(self, x):
        B, L = x.shape
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 1)
        logits = self.lm_head(x)
        return logits

    def extract_embedding(self, x):
        B, L = x.shape
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        pooled = self.global_attn(x)
        subspaces = [proj(pooled) for proj in self.pdm.subspace_proj]
        return F.normalize(torch.cat(subspaces, dim=-1), dim=-1)


class DrugEncoder(nn.Module):
    """Encodes drug molecules (Morgan fingerprints → embedding)."""
    def __init__(self, fp_size=2048, hidden_dim=512, output_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(fp_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, fingerprints):
        return F.normalize(self.encoder(fingerprints), dim=-1)


class CureScoringHead(nn.Module):
    """Takes virus embedding + drug embedding → efficacy score."""
    def __init__(self, virus_dim=384, drug_dim=128):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(virus_dim + drug_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, virus_emb, drug_emb):
        combined = torch.cat([virus_emb, drug_emb], dim=-1)
        return self.scorer(combined).squeeze(-1)


class RabiesDefenseModel(nn.Module):
    """Full model: TCN backbone + PDM for variant understanding + drug scoring."""
    def __init__(self, model_size="pro"):
        super().__init__()
        self.backbone = ViralTCN(model_size=model_size)
        self.model_size = model_size

    def forward(self, x):
        return self.backbone(x)

    def forward_cure(self, virus_seq, drug_fps):
        virus_emb = self.backbone.extract_embedding(virus_seq)
        if not hasattr(self, 'drug_encoder'):
            virus_dim = virus_emb.shape[-1]
            self.drug_encoder = DrugEncoder(output_dim=128).to(virus_emb.device)
            self.cure_head = CureScoringHead(virus_dim=virus_dim, drug_dim=128).to(virus_emb.device)
        drug_emb = self.drug_encoder(drug_fps)
        return self.cure_head(virus_emb, drug_emb)

    def extract_embedding(self, x):
        return self.backbone.extract_embedding(x)
