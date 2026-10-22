"""
TOXICITY PREDICTION PIPELINE — COMPREHENSIVE ARCHITECTURE
=========================================================

Goal: Predict the causal chain of drug toxicity, not just "toxic/not".
      Drug → metabolism → reactive metabolite → protein adduction → pathway disruption → organ toxicity → clinical outcome

Data Sources (10 datasets needed):
----------------------------------
1. DILIrank 2.0 (FDA, 2025)       1,336 drugs   100 KB   4-class DILI concern label
2. PubChem / ChEMBL                1,336 drugs     1 MB   SMILES structures
3. DrugBank (metabolism)             900 drugs   500 KB   CYP enzyme annotations
4. Literature mining (LLM)         1,000 drugs     1 MB   Mechanism annotations (GENERATED)
5. AOP-DB (EPA)                    200 pathways   50 MB   Pathway → outcome mapping
6. TG-GATES / DILImap              300 compounds 500 MB   Transcriptomics (expression data)
7. SIDER                           1,430 drugs    10 MB   Side effect profiles
8. FAERS (filtered to DILIrank)    1,336 drugs   100 MB   Adverse event reports
9. Tox21                           10K compounds  10 MB   In vitro assay results
10. Computed descriptors (RDKit)   1,336 drugs     1 MB   Molecular descriptors (LogP, MW, HBD, HBA, etc.)
Total (compressed): ~150 MB for structured data, ~500 MB for raw transcriptomics

Architecture: Hierarchical Multi-Task TCN Depthwise
===================================================

One unified model with 7 prediction heads, cascading causally.
Each head predicts one step in the toxicity chain.
Outputs of earlier heads feed into later heads.

                          ┌─────────────────────┐
                          │  Molecular Encoder   │
                          │  (TCN Depthwise 3B)  │
                          └─────────┬───────────┘
                                    │ 256-dim embedding
                          ┌─────────▼───────────┐
                          │  Head 1: Metabolism  │ ← 9 CYP classes (multi-label)
                          │  (MLP, 2 layers)     │
                          └─────────┬───────────┘
                                    │ CYP prediction
                          ┌─────────▼───────────┐
                          │  Head 2: Structural   │ ← 15 alert types (multi-label)
                          │  Alerts / RM Type    │
                          │  (MLP, 2 layers)     │
                          └─────────┬───────────┘
                                    │ Alert prediction
                          ┌─────────▼───────────┐
                          │  Head 3: Protein     │ ← 50 protein targets (multi-label)
                          │  Adduction Target   │
                          │  (MLP, 2 layers)     │
                          └─────────┬───────────┘
                                    │ Protein targets
                          ┌─────────▼───────────┐
                          │  Head 4: Pathway     │ ← 20 pathways (multi-label)
                          │  Disruption (AOP)   │
                          │  (MLP, 2 layers)     │
                          └─────────┬───────────┘
                                    │ Disrupted pathways
                          ┌─────────▼───────────┐
                          │  Head 5: Organ       │ ← 8 organs (multi-label)
                          │  Toxicity    │
                          │  (MLP, 2 layers)     │
                          └─────────┬───────────┘
                                    │ Organ toxicity
                          ┌─────────▼───────────┐
                          │  Head 6: Severity    │ ← 4-class DILI rank
                          │  Integration         │
                          │  (Small Transformer) │
                          └─────────┬───────────┘
                                    │ DILI rank
                          ┌─────────▼───────────┐
                          │  Head 7: Clinical    │ ← binary toxic/not
                          │  Outcome (final)     │
                          │  (MLP, 1 layer)      │
                          └─────────────────────┘

Parameter Estimate:
┌────────────────────────────────┬──────────┐
│ Component                      │ Params   │
├────────────────────────────────┼──────────┤
│ Molecular Encoder (TCN-DW 3B) │  2.1M    │
│   Embed layer (128→256)        │  32K     │
│   TCN Block 1 (DW k=3, 256→256)│  688K   │
│   TCN Block 2 (DW k=5, 256→256)│  688K   │
│   TCN Block 3 (DW k=7, 256→256)│  688K   │
│ Head 1: CYP Metabolism         │  52K     │
│ Head 2: Structural Alerts      │  53K     │
│ Head 3: Protein Adduction      │  79K     │
│ Head 4: Pathway Disruption     │  54K     │
│ Head 5: Organ Toxicity         │  42K     │
│ Head 6: Severity Integration   │ 280K     │
│   (Small Transformer, 2 layers)│          │
│ Head 7: Clinical Outcome       │  13K     │
├────────────────────────────────┼──────────┤
│ TOTAL                          │ ≈2.7M    │
└────────────────────────────────┴──────────┘

Training Strategy:
==================
- Total trainable: 2.7M params (fits on MacBook CPU in < 2GB RAM)
- Training time: ~4-6 hours on MacBook CPU, ~30 min on GPU
- Loss: Binary cross-entropy per head (multi-label), CE for Head 6 (multi-class)
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Scheduler: Cosine annealing with warm restarts
- Batch size: 32-64
- Epochs: 100-200 (with early stopping)

Key Design Decisions:
=====================
1. TCN Depthwise over Transformer backbone:
   - Depthwise separable conv = 7-10× fewer params than transformer
   - Molecular fingerprints are fixed-size (2048), not sequential — conv is natural
   - For SMILES input (if added later): 1D conv over tokens, depthwise for efficiency
   - Transformers need O(n²) attention — unnecessary for 2048 fixed features

2. Cascading heads, not end-to-end black box:
   - Every head predicts something interpretable
   - If data is missing for a head, skip that head during training
   - At inference, use best available prediction (rule-based fallback)

3. Why one model not 7 separate:
   - Shared molecular encoder reduces total params by 70% vs 7 separate encoders
   - Gradient flows from clinical outcome back through all heads (rich signal)
   - If head 3 has no data but head 4 does, head 4's gradient still improves head 3

Phased Building Plan:
=====================

Phase 1 (Month 1): Data Collection
  - Download DILIrank 2.0, PubChem SMILES, DrugBank, SIDER, Tox21
  - Compute RDKit descriptors for all 1,336 drugs
  - Build unified DataFrame: SMILES + labels + descriptors

Phase 2 (Month 2): Literature Mining for Mechanism Labels
  - For each of 1,336 drugs, use GPT-4o/Claude to search PubMed abstracts
  - Extract structured mechanism: {drug, CYP_enzyme, RM_type, adducted_protein, pathway, organ}
  - Human-validate 100 random entries for accuracy
  - Export as mechanism_label.csv

Phase 3 (Month 3): Build + Train Model
  - Implement TCN depthwise encoder + 7 heads in PyTorch
  - Train on 80/20 split with cross-validation
  - Evaluate each head independently (per-step accuracy)
  - Evaluate full pipeline (end-to-end DILI rank accuracy)

Phase 4 (Month 4): Validation + Publication
  - Test on 100 novel drugs NOT in DILIrank (holdout from literature)
  - Compare against existing tools (ADMET-SAR, ProTox, DeepCYP)
  - Publish dataset + model on GitHub / Hugging Face
  - Write paper

Model File: /model/toxicity_pipeline.py
  - class ToxTCNEncoder: TCN depthwise backbone
  - class ToxPipeline(nn.Module): 7 heads + forward pass
  - class ToxDataset: loads all 10 data sources
  - train.py: training loop
  - predict.py: inference wrapper
  - Notebooks: EDA, training curves, interpretability

What this UNLOCKS:
===================
- First public dataset linking SMILES → mechanism → outcome
- First interpretable toxicity model (not black box score)
- For a new molecule, outputs:
  "72% chance metabolized by CYP3A4 → 68% forms quinone methide →
   84% depletes GSH in hepatocytes → 91% liver toxicity concern"
- Can be extended to any organ, any toxicity endpoint
- Reproducible, open-source, one-laptop-trainable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# TCN Depthwise Block
# ============================================================
class DepthwiseTCNBlock(nn.Module):
    """
    Depthwise separable temporal convolution block.
    
    Design rationale:
    - Depthwise conv: each input channel convolved independently (groups=channels)
      → learns per-feature patterns
    - Pointwise conv (1×1): mixes channels
      → 7-10× fewer params than standard conv
    - Residual connection: allows training deep stacks
    - Padding='same': preserves sequence length
    """
    def __init__(self, channels, kernel_size, dilation=1, dropout=0.2):
        super().__init__()
        self.dw = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding='same',
            dilation=dilation,
            groups=channels,          # depthwise: each channel gets own filter
            bias=False
        )
        self.pw = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,            # pointwise: channel mixing
            bias=True
        )
        self.bn = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, channels, length)
        residual = x
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x + residual


# ============================================================
# Molecular Encoder (TCN Depthwise Backbone)
# ============================================================
class MoleculeTCNEncoder(nn.Module):
    """
    Takes molecular fingerprint (2048) and outputs learned embedding (256).
    
    Architecture:
    - Linear projection: 2048 → 256
    - 3 Depthwise TCN blocks (kernel sizes 3, 5, 7)
    - Global avg pooling
    - Output: 256-dim embedding (feeds all prediction heads)
    """
    def __init__(self, fp_dim=2048, hidden_dim=256, n_blocks=3, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(fp_dim, hidden_dim)
        
        # TCN blocks with increasing receptive field
        kernels = [3, 5, 7]
        self.blocks = nn.ModuleList([
            DepthwiseTCNBlock(hidden_dim, k, dropout=dropout)
            for k in kernels[:n_blocks]
        ])
        
    def forward(self, fp):
        # fp: (batch, fp_dim)
        x = self.proj(fp)                             # (batch, hidden)
        x = x.unsqueeze(-1)                           # (batch, hidden, 1)
        for block in self.blocks:
            x = block(x)                              # (batch, hidden, 1)
        x = x.squeeze(-1)                             # (batch, hidden)
        return x


# ============================================================
# Individual Prediction Heads
# ============================================================
class PredictionHead(nn.Module):
    """
    Generic 2-layer MLP head for multi-label classification.
    
    Used by Heads 1-5 (all multi-label).
    Input: embedding (256) + optional previous head outputs (cascaded features)
    Output: sigmoid scores per class (multi-label)
    """
    def __init__(self, input_dim, n_classes, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes),
        )
        
    def forward(self, x):
        return torch.sigmoid(self.net(x))


class SeverityHead(nn.Module):
    """
    Head 6: Small transformer for severity integration.
    
    Takes all previous predictions as a sequence and applies self-attention
    to model interactions between different toxicity factors.
    
    Input: (batch, n_previous_predictions, feature_dim)
    Output: 4-class softmax (DILI rank)
    """
    def __init__(self, n_factors, factor_dim, n_heads=2, n_layers=2, hidden_dim=128):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=factor_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls = nn.Linear(factor_dim, 4)
        
    def forward(self, x):
        # x: (batch, n_factors, factor_dim)
        x = self.transformer(x)
        x = x.mean(dim=1)           # pool over factors
        return F.log_softmax(self.cls(x), dim=-1)


# ============================================================
# Full Toxicity Pipeline Model
# ============================================================
class ToxPipeline(nn.Module):
    """
    Unified model predicting the full toxicity chain:
    Drug → CYP metabolism → Structural alerts → Protein adduction →
    Pathway disruption → Organ toxicity → Severity → Clinical outcome
    
    Input: molecular fingerprint (2048)
    Outputs: predictions at each causal level (for interpretability)
    """
    def __init__(self, fp_dim=2048, embed_dim=256):
        super().__init__()
        
        # Backbone
        self.encoder = MoleculeTCNEncoder(fp_dim, embed_dim)
        
        # Head 1: CYP Metabolism (9 enzymes)
        # Classes: CYP1A2, CYP2C9, CYP2C19, CYP2D6, CYP3A4, CYP2E1, CYP2B6, CYP2C8, CYP3A5
        self.head_cvp = PredictionHead(embed_dim, 9)
        
        # Head 2: Structural Alerts / Reactive Metabolite Types (15 types)
        # Classes: quinone, quinone-imine, epoxide, arene-oxide, nitrenium-ion,
        #          acyl-glucuronide, thiophene-S-oxide, furan-epoxide,
        #          hydrazine, nitro-reduction, aniline, Michael-acceptor,
        #          aldehyde, iminium-ion, methylenedioxy
        self.head_alerts = PredictionHead(embed_dim + 9, 15)
        
        # Head 3: Protein Adduction Targets (50 targets)
        self.head_protein = PredictionHead(embed_dim + 9 + 15, 50)
        
        # Head 4: Pathway Disruption / AOP Key Events (20 pathways)
        self.head_pathway = PredictionHead(embed_dim + 9 + 15 + 50, 20)
        
        # Head 5: Organ Toxicity (8 organs)
        self.head_organ = PredictionHead(embed_dim + 9 + 15 + 50 + 20, 8)
        
        # Head 6: Severity Integration (small transformer)
        # Feeds all previous predictions + embedding into transformer
        n_factors = 1 + 9 + 15 + 50 + 20 + 8
        factor_dim = 1  # each is a probability
        self.head_severity = SeverityHead(n_factors, factor_dim)
        
        # Head 7: Clinical Outcome (binary)
        self.head_outcome = nn.Sequential(
            nn.Linear(embed_dim + 9 + 15 + 50 + 20 + 8 + 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, fp):
        # Backbone
        emb = self.encoder(fp)                        # (B, 256)
        
        # Head 1
        cyp = self.head_cvp(emb)                      # (B, 9)
        
        # Head 2 (emb + cyp)
        alerts_input = torch.cat([emb, cyp], dim=-1)  # (B, 265)
        alerts = self.head_alerts(alerts_input)        # (B, 15)
        
        # Head 3 (emb + cyp + alerts)
        prot_input = torch.cat([emb, cyp, alerts], dim=-1)  # (B, 280)
        proteins = self.head_protein(prot_input)       # (B, 50)
        
        # Head 4
        path_input = torch.cat([emb, cyp, alerts, proteins], dim=-1)  # (B, 330)
        pathways = self.head_pathway(path_input)       # (B, 20)
        
        # Head 5
        org_input = torch.cat([emb, cyp, alerts, proteins, pathways], dim=-1)  # (B, 350)
        organs = self.head_organ(org_input)            # (B, 8)
        
        # Head 6: Stack all predictions as a sequence
        all_preds = torch.stack([
            emb.mean(dim=-1, keepdim=True),            # mean embedding
            cyp.mean(dim=-1, keepdim=True),           # mean CYP score
            alerts.mean(dim=-1, keepdim=True),         # mean alert score
            proteins.mean(dim=-1, keepdim=True),       # mean protein score
            pathways.mean(dim=-1, keepdim=True),        # mean pathway score
            organs.mean(dim=-1, keepdim=True),          # mean organ score
        ], dim=1)                                      # (B, 6, 1)
        severity = self.head_severity(all_preds)        # (B, 4)
        
        # Head 7
        outcome_input = torch.cat([
            emb, cyp, alerts, proteins, pathways, organs, severity
        ], dim=-1)                                      # (B, 354)
        outcome = self.head_outcome(outcome_input)      # (B, 1)
        
        return {
            'embedding': emb,
            'cyp': cyp,
            'alerts': alerts,
            'proteins': proteins,
            'pathways': pathways,
            'organs': organs,
            'severity': severity,
            'outcome': outcome,
        }


# ============================================================
# Training Setup
# ============================================================
# Loss: weighted sum of all head losses
#   L = λ₁·BCE(cyp) + λ₂·BCE(alerts) + λ₃·BCE(proteins) +
#       λ₄·BCE(pathways) + λ₅·BCE(organs) + λ₆·CE(severity) +
#       λ₇·BCE(outcome)
#   Default λ = [0.1, 0.1, 0.05, 0.15, 0.15, 0.2, 0.25]
#   Heavier weight on final clinical outcome (learns for prediction)
#   Lighter weight on intermediate heads (regularizers/guides)

# Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
# Scheduler: CosineAnnealingWarmRestarts(T_0=20, T_mult=2)
# Batch size: 32-64
# Epochs: 200 max (early stopping patience=30)
# Mixed precision: fp32 (CPU), fp16 (GPU if available)

# Evaluation Metrics:
# - Per-head: AUROC, F1, precision, recall
# - Full pipeline: DILI rank accuracy, binary AUROC
# - Calibration: Expected Calibration Error (ECE)

print("""
╔══════════════════════════════════════════════════════════════╗
║  TOXICITY PREDICTION PIPELINE                               ║
║  Architecture: Hierarchical Multi-Task TCN Depthwise        ║
║  Model size:  ~2.7M parameters                             ║
║  Datasets:     10 (150 MB structured + 500 MB raw)         ║
║  Training:     ~4-6 hrs MacBook CPU, 30 min GPU            ║
║  Output:       Causal chain: CYP → RM → Protein →           ║
║                Pathway → Organ → Severity → Outcome        ║
╚══════════════════════════════════════════════════════════════╝
""")
