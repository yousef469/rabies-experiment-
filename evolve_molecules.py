"""
Genetic Algorithm Molecule Optimizer

Evolves molecules toward higher composite cure scores.
Mutation + selection + crossover over many generations.

Usage:
    python evolve_molecules.py --generations 50 --population 30
"""
import sys, os, copy, random, json, time, argparse
import numpy as np

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
RDLogger.logger().setLevel(RDLogger.ERROR)

# Seed molecules (known drugs + benzimidazolinone hits)
SEED_SMILES = [
    'CC(=O)Oc1ccccc1C(=O)O',       # Aspirin
    'CN(C)CCCC(c1ccc(F)cc1)c1ccccc1',  # Fluoxetine
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',    # Caffeine
    'CC(=O)NC1=CC=CC=C1',           # Acetaminophen
    'CC(C)Cc1ccc(C(C)C(=O)O)cc1',   # Ibuprofen
    'OC(=O)C1=CC=CC=C1O',           # Salicylic acid
    'CN(C)CC(CO)c1ccc(O)cc1',       # Salbutamol
    'O=c1[nH]c2cccc(-c3ccccc3)c2[nH]1',  # gen_16 (5-Ph benzimidazolinone) ✦
    'O=c1[nH]c2ccc(C#N)cc2[nH]1',       # 5-CN benzimidazolinone ✦
    'O=c1[nH]c2ccc(C(F)(F)F)cc2[nH]1',  # 5-CF3 benzimidazolinone ✦
    'O=c1[nH]c2ccccc2[nH]1',            # parent benzimidazolinone ✦
    'O=c1[nH]c2ccc(Cl)cc2[nH]1',        # 5-Cl
    'O=c1[nH]c2ccc(F)cc2[nH]1',         # 5-F
    'O=c1[nH]c2ccc(OC)cc2[nH]1',        # 5-OMe
    'Cc1cc(=O)[nH]c2ccccc12',           # 4-Methylquinolin-2(1H)-one
    'COc1ccc(C=O)cc1',              # Anisaldehyde
    'c1ccccc1C=O',                  # Benzaldehyde
    'O=c1cc[nH]c(=O)[nH]1',        # Barbituric acid
    'OCc1ccccc1',                   # Benzyl alcohol
    'Oc1ccc(F)cc1',                # 4-Fluorophenol
]

# Termination groups to add during mutation
TERMINALS = ['F', 'Cl', 'Br', 'O', 'N', 'C', 'I']

ATOM_REPLACEMENTS = {
    6: [7, 8, 16],    # C -> N, O, S
    7: [6, 8],         # N -> C, O
    8: [6, 7, 16],    # O -> C, N, S
    16: [6, 8],        # S -> C, O
}


def smi_to_mol(smi):
    try:
        return Chem.MolFromSmiles(smi)
    except:
        return None


def mol_to_smi(mol):
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None


def get_random_atom(mol):
    """Get a random heavy atom index."""
    atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    return random.choice(atoms) if atoms else None


def get_terminal_atoms(mol):
    """Get leaf atoms (degree 1, not H)."""
    return [a.GetIdx() for a in mol.GetAtoms()
            if a.GetDegree() == 1 and a.GetAtomicNum() > 1]


def mutate_add_terminal(mol):
    """Add a terminal group (F, Cl, OH, etc.) to a random atom."""
    mol = copy.deepcopy(mol)
    atom_idx = get_random_atom(mol)
    if atom_idx is None:
        return mol
    rw = Chem.RWMol(mol)
    term = random.choice(TERMINALS)
    term_mol = Chem.MolFromSmiles(term)
    if term_mol is None:
        return mol
    term_atom = term_mol.GetAtomWithIdx(0)
    new_idx = rw.AddAtom(term_atom)
    rw.AddBond(atom_idx, new_idx, Chem.BondType.SINGLE)
    try:
        Chem.SanitizeMol(rw)
        return rw.GetMol()
    except:
        return mol


def mutate_remove_terminal(mol):
    """Remove a random terminal atom (if any)."""
    terminals = get_terminal_atoms(mol)
    if len(terminals) < 2:  # keep at least 2 atoms
        return mol
    atom_idx = random.choice(terminals)
    rw = Chem.RWMol(copy.deepcopy(mol))
    rw.RemoveAtom(atom_idx)
    try:
        Chem.SanitizeMol(rw)
        res = rw.GetMol()
        if res.GetNumAtoms() >= 3:
            return res
    except:
        pass
    return mol


def mutate_replace_atom(mol):
    """Replace a random atom with a different element."""
    mol = copy.deepcopy(mol)
    atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() in ATOM_REPLACEMENTS]
    if not atoms:
        return mol
    atom = random.choice(atoms)
    old_z = atom.GetAtomicNum()
    new_z = random.choice(ATOM_REPLACEMENTS[old_z])
    atom.SetAtomicNum(new_z)
    try:
        Chem.SanitizeMol(mol)
        return mol
    except:
        atom.SetAtomicNum(old_z)  # revert
        return mol


def mutate_change_bond(mol):
    """Toggle a bond between single and double."""
    mol = copy.deepcopy(mol)
    bonds = list(mol.GetBonds())
    random.shuffle(bonds)
    for bond in bonds:
        if bond.IsInRing():
            continue
        bt = bond.GetBondType()
        if bt == Chem.BondType.SINGLE:
            bond.SetBondType(Chem.BondType.DOUBLE)
        elif bt == Chem.BondType.DOUBLE:
            bond.SetBondType(Chem.BondType.SINGLE)
        else:
            continue
        try:
            Chem.SanitizeMol(mol)
            return mol
        except:
            # Revert
            bond.SetBondType(bt)
            continue
    return mol


def mutate_add_ring(mol):
    """Add a ring by bonding two non-adjacent atoms within 4-6 bonds."""
    mol = copy.deepcopy(mol)
    atoms = list(mol.GetAtoms())
    random.shuffle(atoms)
    for a1 in atoms:
        for a2 in atoms:
            if a1.GetIdx() >= a2.GetIdx():
                continue
            if a1.IsInRing() and a2.IsInRing():
                continue
            # Check distance (3-5 bonds apart)
            path = Chem.GetShortestPath(mol, a1.GetIdx(), a2.GetIdx())
            if 3 <= len(path) <= 6:
                rw = Chem.RWMol(mol)
                rw.AddBond(a1.GetIdx(), a2.GetIdx(), Chem.BondType.SINGLE)
                try:
                    Chem.SanitizeMol(rw)
                    return rw.GetMol()
                except:
                    break
    return mol


def crossover(mol1, mol2):
    """Crossover: break both mols at a ring bond and swap."""
    bonds1 = [b for b in mol1.GetBonds() if not b.IsInRing()]
    bonds2 = [b for b in mol2.GetBonds() if not b.IsInRing()]
    if not bonds1 or not bonds2:
        return mol1

    b1 = random.choice(bonds1)
    b2 = random.choice(bonds2)

    try:
        # Fragment molecules
        frags1 = Chem.FragmentOnBonds(mol1, [b1.GetIdx()], addDummies=False)
        frags2 = Chem.FragmentOnBonds(mol2, [b2.GetIdx()], addDummies=False)
        frag_mols1 = Chem.GetMolFrags(frags1, asMols=True, sanitizeFrags=False)
        frag_mols2 = Chem.GetMolFrags(frags2, asMols=True, sanitizeFrags=False)
        if len(frag_mols1) < 2 or len(frag_mols2) < 2:
            return mol1
        # Pick random fragments
        f1 = random.choice(frag_mols1)
        f2 = random.choice(frag_mols2)
        # Combine
        combo = Chem.CombineMols(f1, f2)
        rw = Chem.RWMol(combo)
        rw.AddBond(f1.GetNumAtoms() - 1, f1.GetNumAtoms(), Chem.BondType.SINGLE)
        try:
            Chem.SanitizeMol(rw)
            res = rw.GetMol()
            if res.GetNumAtoms() >= 3:
                return res
        except:
            pass
    except:
        pass
    return mol1


MUTATIONS = [
    mutate_add_terminal,
    mutate_remove_terminal,
    mutate_replace_atom,
    mutate_change_bond,
    mutate_add_ring,
]


def mutate(mol, rate=1.0):
    """Apply 1-2 random mutations to a molecule."""
    if random.random() > rate:
        return mol
    n_ops = random.randint(1, 2)
    for _ in range(n_ops):
        op = random.choice(MUTATIONS)
        mol = op(mol)
    return mol


def score_fitness(smiles, dose_mg=500, n_doses=14, verbose=False):
    """Score a molecule through the multi-dose pipeline.
    
    Uses molar-normalized dose (10 mmol per dose) to avoid MW bias.
    """
    try:
        from pkpd_model import score_candidate_multi_dose
        # Normalize dose by MW: 10 mmol per dose
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mw = Descriptors.MolWt(mol)
        else:
            mw = 300
        dose_normalized = 5.0 * mw  # 5 mmol per dose (consistent molar dose)
        
        result = score_candidate_multi_dose(smiles, name="ga", dose_mg=dose_normalized,
                                             n_doses=n_doses, verbose=verbose)
        return result
    except Exception as e:
        if verbose:
            print(f"  Scoring failed for {smiles}: {e}")
        return None


def score_fitness_light(smiles, dose_mg=500, n_doses=14):
    """Fast fitness: skip viral dynamics, use PK metrics + BBB + toxicity."""
    try:
        from pkpd_model import simulate_pkpd_multi_dose, predict_toxicity, get_bbb_permeability
        pBBB = get_bbb_permeability(smiles)
        pkpd = simulate_pkpd_multi_dose(smiles, dose_mg=dose_mg, n_doses=n_doses)
        tox = predict_toxicity(smiles)

        bbb = pBBB
        trough = pkpd.get('trough_brain_uM', 0)
        cmax = pkpd.get('Cmax_brain_uM', 0)
        time_above = pkpd.get('time_above_ec50_hrs', 0)

        # Composite: BBB + drug exposure + safety
        score = bbb * 0.30 + min(cmax / 100, 1) * 0.25 + min(time_above / 336, 1) * 0.25 + (1 - tox['score']) * 0.20
        return {
            'composite_score': score,
            'pBBB': bbb,
            'trough_brain_uM': trough,
            'Cmax_brain_uM': cmax,
            'time_above_ec50_hrs': time_above,
            'toxicity': tox,
            'viral': {'cured': False, 'survival_fraction': 0},
            'smiles': smiles,
        }
    except:
        return None


# Vina docking cache
_VINA_CACHE = {}
_VINA_RECEPTOR_READY = False


def score_fitness_vina(smiles, use_immune_model=True, verbose=False):
    """Fitness with Vina docking score + PK/PD/BVB + immune viral dynamics.

    ~20s per call (Vina docking ~15s + PK/PD ~5s).
    """
    try:
        from vina_dock import dock_smiles, pdb_to_pdbqt, get_pocket_center
        from pkpd_model import simulate_pkpd_multi_dose, simulate_viral_dynamics_immune, predict_toxicity, get_bbb_permeability
        import os

        # Cached docking
        global _VINA_CACHE, _VINA_RECEPTOR_READY
        if smiles in _VINA_CACHE:
            vina_score = _VINA_CACHE[smiles]
        else:
            if not _VINA_RECEPTOR_READY:
                os.makedirs('/tmp/vina_ga', exist_ok=True)
                pdb_to_pdbqt('pdb_cache/6lgx.pdb', '/tmp/vina_ga/receptor.pdbqt')
                _VINA_RECEPTOR_READY = True
            center = get_pocket_center('pdb_cache/6lgx.pdb')
            score, _ = dock_smiles(smiles, '/tmp/vina_ga/receptor.pdbqt', center,
                                    name=f'ga_{len(_VINA_CACHE)}', work_dir='/tmp/vina_ga')
            vina_score = score if score is not None else -5.0
            _VINA_CACHE[smiles] = vina_score

        dock_norm = max(0, min(1, (-vina_score - 5) / 5))

        # PK/PD (QD for differentiation; TID + immune cures everything)
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        mw = Descriptors.MolWt(mol) if mol else 300
        dose = 5e-3 * mw * 1000  # 5 mmol daily
        pkpd = simulate_pkpd_multi_dose(smiles, dose_mg=dose, n_doses=14, interval_hrs=24)

        # Viral dynamics with immune response (QD keeps trough below EC50 → differentiates)
        viral = simulate_viral_dynamics_immune(pkpd, duration_hrs=720, treatment_delay_hrs=48)
        survival = viral['survival_fraction']
        cured = viral.get('cured', False)

        # BBB + toxicity
        bbb = get_bbb_permeability(smiles)
        tox = predict_toxicity(smiles)
        tox_safe = 1 - tox.get('score', 1)

        # Composite: docking + viral survival + BBB + safety
        composite = dock_norm * 0.35 + survival * 0.35 + bbb * 0.15 + tox_safe * 0.15

        if verbose:
            print(f"  dock={vina_score:.2f}({dock_norm:.2f}) surv={survival:.3f} bbb={bbb:.2f} tox={tox_safe:.2f} => {composite:.3f}")

        return {
            'composite_score': composite,
            'vina_score': vina_score,
            'dock_norm': dock_norm,
            'pBBB': bbb,
            'trough_brain_uM': pkpd.get('trough_brain_uM', 0),
            'Cmax_brain_uM': pkpd.get('Cmax_brain_uM', 0),
            'viral': viral,
            'toxicity': tox,
            'survival': survival,
            'cured': cured,
            'smiles': smiles,
        }
    except Exception as e:
        if verbose:
            print(f"  Vina scoring failed: {e}")
        return None


def is_valid_mol(smiles):
    """Check if SMILES is valid and drug-like."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        mw = Descriptors.MolWt(mol)
        if mw < 100 or mw > 600:
            return False
        num_atoms = mol.GetNumAtoms()
        if num_atoms < 5 or num_atoms > 80:
            return False
        return True
    except:
        return False


def evolve(generations=50, population_size=30, dose_mg=500, n_doses=14, verbose=True, resume_from=None, use_vina=False):
    """Run genetic algorithm to evolve molecules toward higher cure scores."""
    random.seed(42)
    np.random.seed(42)

    if use_vina:
        print(f"\n  VINA MODE: ~20s per molecule, ~{20*population_size*generations//60}min total")
        print(f"  Preparing Vina receptor...")
        try:
            from vina_dock import pdb_to_pdbqt, get_pocket_center
            import os
            os.makedirs('/tmp/vina_ga', exist_ok=True)
            pdb_to_pdbqt('pdb_cache/6lgx.pdb', '/tmp/vina_ga/receptor.pdbqt')
            center = get_pocket_center('pdb_cache/6lgx.pdb')
            print(f"  Docking center: {center}")
        except Exception as e:
            print(f"  Vina init failed: {e}, falling back to heuristic")
            use_vina = False

    # Initialize population
    if resume_from:
        with open(resume_from) as f:
            data = json.load(f)
        population = [(Chem.MolFromSmiles(s), s) for s in data['smiles']]
        population = [(m, s) for m, s in population if m is not None]
        start_gen = data.get('generation', 0) + 1
        print(f"Resumed from generation {start_gen}")
    else:
        population = []
        for smi in SEED_SMILES:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                population.append((mol, smi))
        # Fill with mutants of seeds
        while len(population) < population_size:
            parent = random.choice(population[:min(10, len(population))])
            child = mutate(parent[0])
            csmi = mol_to_smi(child)
            if csmi and csmi not in [p[1] for p in population]:
                population.append((child, csmi))
        start_gen = 0

    best_score = -1
    best_smiles = None
    history = []

    for gen in range(start_gen, generations + 1):
        if verbose:
            print(f"\n{'='*60}")
            print(f"  GENERATION {gen}/{generations}  (pop={len(population)})")
            print(f"{'='*60}")

        # Score all
        scored = []
        for i, (mol, smi) in enumerate(population):
            if use_vina:
                result = score_fitness_vina(smi, verbose=False)
            else:
                result = score_fitness(smi, dose_mg=dose_mg, n_doses=n_doses, verbose=False)
            if result is None:
                score = 0
                cured = False
                survival = 0
            else:
                score = result['composite_score']
                cured = result['viral']['cured']
                survival = result['viral']['survival_fraction']

            scored.append((smi, score, cured, survival, result))
            if verbose and (i < 3 or score > 0.6 or cured):
                marker = "**CURE**" if cured else " "
                if use_vina:
                    v = result.get('vina_score', 0)
                    print(f"  {i+1:>2}. {marker} {smi:<40} score={score:.4f}  surv={survival:.3f}  dock={v:.2f}")
                else:
                    print(f"  {i+1:>2}. {marker} {smi:<40} score={score:.4f}  surv={survival:.4f}")

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        gen_best = scored[0][1]
        gen_best_smi = scored[0][0]
        gen_best_cured = scored[0][2]

        if gen_best > best_score:
            best_score = gen_best
            best_smiles = gen_best_smi

        history.append({
            'generation': int(gen),
            'best_score': float(gen_best),
            'best_smiles': str(gen_best_smi),
            'cured': bool(gen_best_cured),
            'pop_size': int(len(population)),
        })

        print(f"\n  Generation {gen} best: {gen_best_smi}  score={gen_best:.4f}  cured={gen_best_cured}")
        print(f"  All-time best: {best_smiles}  score={best_score:.4f}")

        if gen_best_cured:
            print(f"\n  *** CURE FOUND in generation {gen}! ***")
            print(f"  SMILES: {gen_best_smi}")
            # Save immediately
            with open('cure_found.json', 'w') as f:
                json.dump({'generation': gen, 'smiles': gen_best_smi, 'score': gen_best,
                          'result': scored[0][4]}, f, indent=2, default=str)
            # Break or continue? Let's continue to see if we can find better
            if verbose:
                print("  Continuing to find better cure candidates...")

        # Selection: keep top 30%
        keep = max(population_size // 3, 5)
        survivors = [s for s in scored[:keep] if is_valid_mol(s[0])]

        if not survivors:
            print("  No valid survivors! Restarting from seeds...")
            survivors = [(smi_to_mol(s), 0, False, 0, None) for s in SEED_SMILES[:5]]
            survivors = [(m, s, 0, False, None) for m, s in survivors if m is not None]

        # Create next generation
        next_pop = []
        for smi, _, _, _, _ in survivors:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                next_pop.append((mol, smi))

        # Fill rest with mutations of top survivors
        attempts = 0
        max_attempts = population_size * 10
        while len(next_pop) < population_size and attempts < max_attempts:
            attempts += 1
            parent_mol, parent_smi = random.choice(next_pop[:max(3, len(next_pop)//2)])
            child_mol = mutate(parent_mol)
            child_smi = mol_to_smi(child_mol)
            if child_smi and is_valid_mol(child_smi) and child_smi not in [p[1] for p in next_pop]:
                next_pop.append((child_mol, child_smi))

        # Add crossover children
        n_cross = population_size // 5
        for _ in range(n_cross):
            p1_mol, _ = random.choice(next_pop[:max(5, len(next_pop)//2)])
            p2_mol, _ = random.choice(next_pop[:max(5, len(next_pop)//2)])
            child = crossover(p1_mol, p2_mol)
            csmi = mol_to_smi(child)
            if csmi and is_valid_mol(csmi) and csmi not in [p[1] for p in next_pop]:
                next_pop.append((child, csmi))

        # If still too small, add mutants of best
        while len(next_pop) < population_size:
            p_mol, _ = random.choice(next_pop[:max(3, len(next_pop)//3)])
            child = mutate(p_mol, rate=1.5)
            csmi = mol_to_smi(child)
            if csmi and csmi not in [p[1] for p in next_pop]:
                next_pop.append((child, csmi))

        population = next_pop[:population_size]

        # Save checkpoint
        checkpoint = {
            'generation': int(gen),
            'best_score': float(best_score),
            'best_smiles': str(best_smiles),
            'smiles': [str(s[0]) for s in scored[:10]],
            'history': history[-10:],
        }
        with open('ga_checkpoint.json', 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)

    # Final results
    if verbose:
        print(f"\n{'='*60}")
        print(f"  EVOLUTION COMPLETE")
        print(f"{'='*60}")
        print(f"  Generations: {generations}")
        print(f"  Best score:  {best_score:.4f}")
        print(f"  Best SMILES: {best_smiles}")
        print(f"  Cured:       {bool(best_score > 0.8)}")

    if best_smiles and False:  # skip final detailed scoring (already done)
        pass

    results = {
        'generations': generations,
        'best_score': best_score,
        'best_smiles': best_smiles,
        'history': history,
    }
    with open('ga_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generations', type=int, default=30)
    parser.add_argument('--population', type=int, default=24)
    parser.add_argument('--dose-mg', type=int, default=500)
    parser.add_argument('--n-doses', type=int, default=14)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-vina', action='store_true', help='Use Vina docking score in fitness')
    args = parser.parse_args()

    random.seed(args.seed)

    if args.use_vina:
        args.population = min(args.population, 12)
        args.generations = min(args.generations, 15)
        print(f"  Vina mode: pop={args.population}, gen={args.generations}")

    print(f"Genetic Algorithm Molecule Optimizer")
    print(f"  Generations: {args.generations}")
    print(f"  Population:  {args.population}")
    print(f"  Dose:        {args.dose_mg}mg x {args.n_doses} days (IV)")
    print(f"  Mutations:   {len(MUTATIONS)} types")
    print(f"  Seeds:       {len(SEED_SMILES)} molecules")
    print(f"  Fitness:     {'Vina docking + PK/PD + immune' if args.use_vina else 'Heuristic PK/PD'}")

    results = evolve(
        generations=args.generations,
        population_size=args.population,
        dose_mg=args.dose_mg,
        n_doses=args.n_doses,
        resume_from=args.resume,
        use_vina=args.use_vina,
    )
