"""
Multi-virus protein encoder configuration.
Amino acid vocabulary + hyperparameters for ViroGym training.
"""
import torch

# ── Amino acid vocabulary ──────────────────────────────────────────────────
AA_VOCAB = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
AA_PAD_TOKEN = 0
AA_VOCAB_SIZE = len(AA_VOCAB) + 1  # 21 (0=pad, 1-20=AAs)
AA_MAX_SEQ_LEN = 200   # truncated for CPU speed (covers most viral RBDs/domains)

# ── Virus type encoding ────────────────────────────────────────────────────
VIRUS_TYPES = {
    "RABV": 0, "SCV2": 1, "HIV": 2, "IAV": 3, "ZIKV": 4,
    "DEN": 5, "NIPAH": 6, "LASV": 7, "HBV": 8, "HCV": 9,
    "AAV2": 10, "CXB3": 11, "PESV": 12,
}
NUM_VIRUS_TYPES = len(VIRUS_TYPES)

# ── Phenotype categories (from ViroGym benchmark.csv) ──────────────────────
PHENOTYPE_CATEGORIES = {
    "binding": 0, "cell_entry": 1, "expression": 2,
    "immune_escape": 3, "viral_growth": 4, "stability": 5,
}
NUM_PHENOTYPES = len(PHENOTYPE_CATEGORIES)

# ── Model architecture ─────────────────────────────────────────────────────
EMBED_DIM = 128       # tiny for CPU
NUM_BLOCKS = 3        # fewer blocks
KERNEL_SIZE = 5
EXPAND_RATIO = 2
DROPOUT = 0.1
NUM_SUBSPACES = 2
SUBSPACE_DIM = 32
PDM_TEMPERATURE = 0.07

# Drug encoder (mutation encoder) dimensions
MUTATION_EMBED_DIM = 128  # encodes (position, wildtype_aa, mutant_aa)
CURE_SCORE_DIM = 64       # intermediate fusion dim

# ── Training ────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 15
MAX_SAMPLES = 50000    # subsample for CPU speed
LR = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
GRAD_CLIP = 1.0

DEVICE = "cpu"
NUM_WORKERS = 0
SEED = 42
