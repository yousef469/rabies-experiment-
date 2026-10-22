"""
Protein encoder for multi-virus DMS prediction.
Same DepthwiseSeparableConv1D + PDM architecture as ViralTCN, adapted for amino acids.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from protein_config import (
    AA_VOCAB_SIZE, EMBED_DIM, NUM_BLOCKS, KERNEL_SIZE,
    EXPAND_RATIO, DROPOUT, NUM_SUBSPACES, SUBSPACE_DIM,
    PDM_TEMPERATURE, NUM_VIRUS_TYPES, NUM_PHENOTYPES,
    MUTATION_EMBED_DIM, CURE_SCORE_DIM,
)


class DepthwiseSeparableConv1DBlock(nn.Module):
    def __init__(self, dim, kernel_size=9, dilation=1, expand_ratio=2, dropout=0.1):
        super().__init__()
        hidden_dim = dim * expand_ratio
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                                   padding=padding, dilation=dilation, groups=dim, bias=False)
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
        x_perm = x.permute(0, 2, 1)
        q = self.query(x_perm)
        k = self.key(x_perm)
        v = self.value(x_perm)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return out.mean(dim=1)


class PrototypeDistributionModule(nn.Module):
    def __init__(self, input_dim, num_subspaces=4, subspace_dim=32,
                 num_classes=NUM_PHENOTYPES, temperature=PDM_TEMPERATURE):
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


class MutationEncoder(nn.Module):
    """Encodes amino acid mutations as 'drug-like' embeddings.
    Input: (batch, 3) tensor of [position, wildtype_aa_idx, mutant_aa_idx]
    Output: (batch, output_dim) embedding
    """
    def __init__(self, vocab_size=21, embed_dim=64, output_dim=MUTATION_EMBED_DIM):
        super().__init__()
        self.pos_embed = nn.Embedding(2000, embed_dim)  # max position
        self.wt_embed = nn.Embedding(vocab_size, embed_dim)
        self.mt_embed = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 3, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, mutation_tensor):
        pos = self.pos_embed(mutation_tensor[:, 0].clamp(0, 1999))
        wt = self.wt_embed(mutation_tensor[:, 1].clamp(0, 20))
        mt = self.mt_embed(mutation_tensor[:, 2].clamp(0, 20))
        combined = torch.cat([pos, wt, mt], dim=-1)
        return F.normalize(self.proj(combined), dim=-1)


class ProteinEncoder(nn.Module):
    """Full protein encoder: AA embedding → TCN blocks → attention → PDM.
    Outputs phenotype classification logits + protein embedding.
    """
    def __init__(self, embed_dim=EMBED_DIM, num_blocks=NUM_BLOCKS):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(AA_VOCAB_SIZE, embed_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, 1))

        self.blocks = nn.ModuleList([
            DepthwiseSeparableConv1DBlock(
                dim=embed_dim, kernel_size=KERNEL_SIZE,
                dilation=2 ** i, expand_ratio=EXPAND_RATIO, dropout=DROPOUT,
            )
            for i in range(num_blocks)
        ])

        self.global_attn = GlobalAttention1D(embed_dim)
        self.pdm = PrototypeDistributionModule(
            input_dim=embed_dim, num_subspaces=NUM_SUBSPACES,
            subspace_dim=SUBSPACE_DIM, num_classes=NUM_PHENOTYPES,
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2 and 'norm' not in name and 'subspace_proj' not in name:
                if 'embedding' in name:
                    nn.init.normal_(param, mean=0.0, std=0.02)
                elif 'depthwise' in name:
                    nn.init.normal_(param, mean=0.0, std=0.01)
                elif 'expand' in name or 'project' in name:
                    nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        pooled = self.global_attn(x)
        logits, embedding = self.pdm(pooled)
        return logits, embedding

    def extract_embedding(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        pooled = self.global_attn(x)
        subspaces = [proj(pooled) for proj in self.pdm.subspace_proj]
        return F.normalize(torch.cat(subspaces, dim=-1), dim=-1)


class DMSHead(nn.Module):
    """Predicts DMS score from protein embedding + mutation encoding.
    Combines virus understanding with mutation effect prediction.
    """
    def __init__(self, protein_dim=NUM_SUBSPACES * SUBSPACE_DIM,
                 mutation_dim=MUTATION_EMBED_DIM):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(protein_dim + mutation_dim, CURE_SCORE_DIM * 2),
            nn.LayerNorm(CURE_SCORE_DIM * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(CURE_SCORE_DIM * 2, CURE_SCORE_DIM),
            nn.LayerNorm(CURE_SCORE_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(CURE_SCORE_DIM, 1),
        )

    def forward(self, protein_emb, mutation_emb):
        combined = torch.cat([protein_emb, mutation_emb], dim=-1)
        return self.scorer(combined).squeeze(-1)


class MultiVirusModel(nn.Module):
    """Complete model: ProteinEncoder + MutationEncoder + DMSHead.
    Trained end-to-end on ViroGym DMS data.
    """
    def __init__(self):
        super().__init__()
        self.protein_encoder = ProteinEncoder()
        self.mutation_encoder = MutationEncoder()
        self.dms_head = DMSHead()

    def forward(self, seq_tensor, mutation_tensor):
        protein_emb = self.protein_encoder.extract_embedding(seq_tensor)
        mutation_emb = self.mutation_encoder(mutation_tensor)
        return self.dms_head(protein_emb, mutation_emb)

    def forward_phenotype(self, seq_tensor):
        return self.protein_encoder(seq_tensor)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
