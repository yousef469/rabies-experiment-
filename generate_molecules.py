"""
Molecule generator: scaffold decoration using RDKit's ReplaceSubstructs.
Generates novel antiviral candidates, scores with trained DrugEncoder.
"""
import json, os, sys, torch, numpy as np, warnings
warnings.filterwarnings('ignore')
from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors

# Known antiviral scaffolds with [*:1] attachment point
SCAFFOLDS = [
    Chem.MolFromSmiles("c1ccc2[nH]c(=O)[nH]c2c1[*:1]"),       # Quinazolinone
    Chem.MolFromSmiles("c1cnc2ccccc2n1[*:1]"),                   # Quinoxaline
    Chem.MolFromSmiles("c1ccc2[nH]cnc2c1[*:1]"),                 # Benzimidazole
    Chem.MolFromSmiles("c1ccncc1[*:1]"),                         # Pyridine
    Chem.MolFromSmiles("c1cnc(N)nc1[*:1]"),                      # Pyrimidine
    Chem.MolFromSmiles("C1CCNCC1[*:1]"),                          # Piperidine
    Chem.MolFromSmiles("C1COCCN1[*:1]"),                          # Morpholine
    Chem.MolFromSmiles("c1cnc2ccccc2n1[*:1]"),                   # Quinoline
    Chem.MolFromSmiles("O=C1CCCN1[*:1]"),                        # 2-Piperidone
    Chem.MolFromSmiles("C1CC(=O)NC1[*:1]"),                      # Pyrrolidinone
    Chem.MolFromSmiles("c1ccc(F)cc1[*:1]"),                      # Fluorobenzene
    Chem.MolFromSmiles("c1ccc(Cl)cc1[*:1]"),                     # Chlorobenzene
]

# Simple R-groups (valid SMILES fragments)
R_GROUPS = [
    "N",           # amino
    "O",           # hydroxyl
    "NC(=O)C",     # acetamide
    "C(=O)N",      # carboxamide
    "C(=O)O",      # carboxylic acid
    "C(=O)OC",     # methyl ester
    "S(=O)(=O)N",  # sulfonamide
    "OCCN",        # ethanolamine
    "OCC",         # ethoxy
    "N(C)C",       # dimethylamine
    "C#N",         # nitrile
    "F",           # fluoro
    "Cl",          # chloro
    "Br",          # bromo
    "CC",          # ethyl
    "CCC",         # propyl
    "c1ccccc1",    # phenyl
    "C2CC2",       # cyclopropyl
    "OCCOC",       # PEG-like
]

# Pre-built novel antiviral-like molecules
NOVEL_MOLECULES = [
    "c1ccc2[nH]c(=O)[nH]c2c1NC(=O)N",              # Quinazolinone urea
    "c1ccc2[nH]c(=O)[nH]c2c1NC(=O)CC",             # Quinazolinone propionamide
    "c1ccc2[nH]c(=O)[nH]c2c1NC(=O)c1ccccc1",       # Quinazolinone benzamide
    "c1ccncc1NC(=O)c1ccc(F)cc1",                     # Pyridine-fluorobenzamide
    "c1ccncc1NC(=O)c1ccc(Cl)cc1",                    # Pyridine-chlorobenzamide
    "c1cnc(NC(=O)c2ccc(F)cc2)nc1",                   # Aminopyrimidine-fluorobenzamide
    "C1CCN(C(=O)c2ccc(F)cc2)CC1",                    # Piperidine-fluorobenzamide
    "C1COCC(NC(=O)c2ccc(F)cc2)C1",                   # Morpholine-fluorobenzamide
    "c1ccc2[nH]cnc2c1NC(=O)C(F)(F)F",               # Benzimidazole-trifluoroacetamide
    "c1ccc2[nH]cnc2c1NC(=O)c1ccncc1",               # Benzimidazole-nicotinamide
    "c1ccncc1NC(=O)C(F)(F)F",                        # Pyridine-trifluoroacetamide
    "c1ccncc1NC(=O)C1CC1",                           # Pyridine-cyclopropanecarboxamide
    "c1ccncc1NC(=O)C1CCCC1",                         # Pyridine-cyclopentanecarboxamide
    "O=C(Nc1ccncc1)c1ccc(F)cc1",                     # Nicotinamide-fluorobenzene
    "O=C(Nc1ccncc1)c1ccc(Cl)cc1",                    # Nicotinamide-chlorobenzene
    "O=C(Nc1ccncc1)c1ccc(O)cc1",                     # Nicotinamide-hydroxybenzene
    "O=C(Nc1ccncc1)C(F)(F)F",                        # Nicotinamide-trifluoromethyl
    "O=C(Nc1ccncc1)C1CC1",                           # Nicotinamide-cyclopropyl
    "c1ccc2[nH]c(=O)[nH]c2c1NC(=O)C(F)(F)F",       # Quinazolinone-CF3
    "c1ccc2[nH]c(=O)[nH]c2c1NC(=O)C1CC1",           # Quinazolinone-cyclopropyl
    "c1ccc2[nH]c(=O)[nH]c2c1S(=O)(=O)N",            # Quinazolinone-sulfonamide
    "c1ccncc1S(=O)(=O)N",                            # Pyridine-sulfonamide
    "c1ccncc1OCCN",                                   # Pyridine-ethanolamine
    "O=C(Nc1ccc(F)cc1)c1ccncc1",                     # Fluoroaniline-nicotinamide
    "O=C(Nc1ccc(Cl)cc1)c1ccncc1",                    # Chloroaniline-nicotinamide
    "O=C(NC1CC1)c1ccncc1",                           # Cyclopropyl-nicotinamide
    "O=C(NC1CCCC1)c1ccncc1",                         # Cyclopentyl-nicotinamide
    "c1ccc2[nH]cnc2c1NC(=O)c1ccc(N)cc1",            # Benzimidazole-aminobenzamide
    "c1ccncc1NC(=O)c1ccc(N)cc1",                     # Pyridine-aminobenzamide
    "c1ccncc1NC(=O)c1ccc(O)cc1",                     # Pyridine-hydroxybenzamide
    "C1CCN(C(=O)C(F)(F)F)CC1",                      # Piperidine-CF3
    "C1CCN(C(=O)c1ccc(N)cc1)CC1",                    # Piperidine-aminobenzamide
    "C1COCC(C(=O)Nc2ccncc2)C1",                      # Morpholine-nicotinamide
    "c1ccc2[nH]c(=O)[nH]c2c1NCCN",                  # Quinazolinone-ethylenediamine
    "O=C(Nc1ccncc1)c1ccc(C#N)cc1",                  # Nicotinamide-benzonitrile
    "c1ccncc1NC(=O)C1CCNCC1",                        # Pyridine-piperidineamide
    "c1ccncc1NC(=O)C1COCC1",                         # Pyridine-tetrahydropyranamide
    "c1ccc2[nH]c(=O)[nH]c2c1NC(=O)c1ccc(O)cc1",    # Quinazolinone-hydroxybenzamide
    "c1ccc2[nH]c(=O)[nH]c2c1NC(=O)c1ccc(N)cc1",    # Quinazolinone-aminobenzamide
    "c1ccncc1NC(=O)c1ccc(C#N)cc1",                  # Pyridine-cyanobenzamide
    "c1ccncc1NC(=O)c1ccncc1",                        # Bis-pyridine amide
    "c1ccc2[nH]c(=O)[nH]c2c1NC(=O)c1ccncc1",       # Quinazolinone-nicotinamide
    "c1ccncc1NC(=O)c1ccc(Cl)c(Cl)c1",               # Pyridine-dichlorobenzamide
    "c1ccc2[nH]c(=O)[nH]c2c1NC(=O)c1ccc(Br)cc1",   # Quinazolinone-bromobenzamide
    "c1ccncc1NC(=O)c1ccc(Br)cc1",                    # Pyridine-bromobenzamide
    "O=C(Nc1ccncc1)c1ccc(I)cc1",                     # Nicotinamide-iodobenzene
    "c1ccncc1NC(=O)C(F)(F)F",                        # Pyridine-CF3 amide
    "c1ccncc1NC(=O)C(CC)CC",                         # Pyridine-ethylbutyramide
    "c1ccncc1NC(=O)c1ccccc1N",                       # Pyridine-anthranilamide
    "c1ccc2[nH]c(=O)[nH]c2c1NC(=O)c1ccccc1Cl",     # Quinazolinone-chlorobenzamide
]


def enumerate_molecules(max_mols=300):
    """Generate molecules using RDKit attachment point replacement + pre-built set."""
    generated = []
    seen = set()

    for scaffold_mol in SCAFFOLDS:
        if scaffold_mol is None:
            continue
        for rg_smiles in R_GROUPS:
            rg_mol = Chem.MolFromSmiles(rg_smiles)
            if rg_mol is None:
                continue
            try:
                # Add attachment point to R-group
                rw = Chem.RWMol(rg_mol)
                rw.AddAtom(Chem.Atom(0))  # dummy atom
                rw.AddBond(0, rw.GetNumAtoms() - 1, Chem.BondType.SINGLE)
                rg_with_attach = Chem.MolToSmiles(rw)

                # Replace attachment point
                patt = Chem.MolFromSmiles("[*:1]")
                new_mol = Chem.ReplaceSubstructs(scaffold_mol, patt, rg_mol)[0]
                smi = Chem.MolToSmiles(new_mol)
                if smi not in seen:
                    seen.add(smi)
                    generated.append(smi)
            except:
                continue

    # Add pre-built molecules
    for smi in NOVEL_MOLECULES:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            canonical = Chem.MolToSmiles(mol)
            if canonical not in seen:
                seen.add(canonical)
                generated.append(canonical)

    return generated[:max_mols]


def score_molecules(smiles_list, top_k=20):
    """Score with trained DrugEncoder + CureScoringHead."""
    sys.path.insert(0, os.path.dirname(__file__))
    from screen_drugs_real import smiles_to_morgan, DrugScreeningModel, DEVICE

    model = DrugScreeningModel('checkpoints_multivirus/best_pearson.pt', 'checkpoints_multivirus/best_cure_head.pt')

    seq = ('MNIQFIINPIYAWKIMGMALILIVVLVNLVLIVTITLFVFYEGEDPTTYSYKGIFNCHTQNYRLSSTNYCPRSSSGGNR'
           'HCAYNANKHQYSVVITHYGNMLHTIKETGFNLDIITVDSSSFSNHTASFNYSVVDLIIREHYVYNAITKNLHESGLYI'
           'ATRGTYSCSVSQKLYYDYGHRSSLHLLLNDGLVEDTGLSYPLSKNYFETKITYFISAITLIIILIIILIIILKRRHRRHRRRH')

    fps, valid = [], []
    for smi in smiles_list:
        fp = smiles_to_morgan(smi)
        if fp is not None:
            fps.append(fp)
            valid.append(smi)

    fps_tensor = torch.tensor(np.array(fps), dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        scores = model.score_drugs(seq, fps_tensor)

    scored = []
    for smi, score in zip(valid, scores.cpu().numpy()):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            scored.append({
                'smiles': smi,
                'score': float(score),
                'mw': round(Descriptors.MolWt(mol), 1),
                'logp': round(Descriptors.MolLogP(mol), 2),
                'tpsa': round(Descriptors.TPSA(mol), 1),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
            })
    scored.sort(key=lambda x: x['score'], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    print("Generating novel antiviral candidates...")
    molecules = enumerate_molecules(max_mols=300)

    druglike = []
    for smi in molecules:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            if 150 <= mw <= 500 and -2 <= logp <= 5:
                druglike.append(smi)
    print(f"{len(druglike)} drug-like molecules generated")

    print("Scoring with trained DrugEncoder + CureScoringHead...")
    top = score_molecules(druglike, top_k=20)

    print()
    print("=" * 80)
    print("Top 20 Novel Antiviral Candidates for Rabies (Generated + Scored)")
    print("=" * 80)
    print(f"{'Rank':<5} {'Score':<9} {'MW':<8} {'LogP':<7} SMILES")
    print("-" * 80)
    for i, m in enumerate(top, 1):
        print(f"{i:<5} {m['score']:<9.4f} {m['mw']:<8} {m['logp']:<7} {m['smiles']}")

    with open("generated_molecules.json", "w") as f:
        json.dump(top, f, indent=2)
    print(f"\nSaved to generated_molecules.json")
