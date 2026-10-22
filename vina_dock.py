"""
Molecular docking with AutoDock Vina for rabies G protein.
Converts PDB + SMILES → PDBQT → Vina docking → binding scores.

Usage:
    python vina_dock.py --receptor pdb_cache/6lgx.pdb --smiles "CC(=O)Oc1ccccc1C(=O)O"
    python vina_dock.py --batch generated_molecules.json --top-n 20
"""
import sys, os, json, subprocess, tempfile, math, re
import numpy as np

HAS_RDKIT = False
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    HAS_RDKIT = True
except:
    pass

HAS_BIO = False
try:
    from Bio.PDB import PDBParser, PDBIO
    from Bio.PDB.Polypeptide import PPBuilder
    HAS_BIO = True
except:
    pass

VINA_BIN = "/usr/local/bin/vina"
VINA_SPLIT_BIN = "/usr/local/bin/vina_split"

# Pocket residues for 6LGX (rabies G protein chain A, ~377 residues)
# These correspond to the known epitope regions
POCKET_RESIDUES = list(range(240, 260)) + list(range(100, 120))  # adjust based on structure

# Simple AutoDock atom types based on element
ATOM_TYPE_TABLE = {
    1: 'H',    # hydrogen
    6: 'C',    # carbon
    7: 'N',    # nitrogen
    8: 'O',    # oxygen
    16: 'S',   # sulfur
    9: 'F',   # fluorine
    17: 'Cl',  # chlorine
    35: 'Br',  # bromine
    53: 'I',   # iodine
    15: 'P',   # phosphorus
}


def pdb_to_pdbqt(pdb_path, output_path, chain_id='A'):
    """Convert PDB to PDBQT using Biopython + simple atom typing.
    
    Writes fixed-width PDBQT format compatible with AutoDock Vina.
    Does NOT add hydrogens (Vina handles this internally).
    """
    if not HAS_BIO:
        print("Biopython not available")
        return False

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('receptor', pdb_path)
    except Exception as e:
        print(f"  PDB parse error: {e}")
        return False

    model = structure[0]
    if chain_id not in model:
        print(f"  Chain {chain_id} not found, using first available")
        chain_id = list(model.child_dict.keys())[0]

    chain = model[chain_id]

    # Standard amino acid one-letter codes
    AA_CODES = {'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS',
                 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL',
                 'TRP', 'TYR'}

    with open(output_path, 'w') as f:
        f.write("REMARK  PDBQT generated from PDB\n")
        atom_serial = 0
        for residue in chain:
            resname = residue.get_resname().strip()
            if resname not in AA_CODES:
                continue  # skip waters, ligands
            resid = residue.get_id()[1]
            chain_id_out = chain.get_id()
            for atom in residue:
                atom_serial += 1
                if atom.get_full_id()[3][0].strip():
                    continue  # skip alternate conformations
                elem = atom.element.strip() if atom.element else atom.get_name()[0]
                if not elem:
                    continue
                x, y, z = atom.get_coord()

                atype = ATOM_TYPE_TABLE.get({
                    'H': 1, 'C': 6, 'N': 7, 'O': 8,
                    'S': 16, 'F': 9, 'CL': 17, 'BR': 35, 'I': 53, 'P': 15
                }.get(elem.upper(), 6), 'C')

                # Assign proper PDB atom name
                atom_name = atom.get_name().strip()
                # Pad atom name to 4 chars
                if len(atom_name) < 4:
                    atom_name = f" {atom_name:<3}"

                # Fixed-width PDBQT format (matching meeko's layout)
                line = f"ATOM  {atom_serial:>5} {atom_name:<4} {resname:<3} {chain_id_out}{resid:>4}    "
                line += f"{x:8.3f}{y:8.3f}{z:8.3f}"
                line += f"  1.00  0.00"
                line += f"  {0.0:8.4f} {atype}\n"
                f.write(line)
        f.write("TER\nEND\n")

    return True


def smiles_to_pdbqt(smiles, output_path, name="ligand"):
    """Convert SMILES to PDBQT using meeko."""
    if not HAS_RDKIT:
        return False

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    mol = Chem.AddHs(mol)
    try:
        embed_result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if embed_result == 0:
            AllChem.MMFFOptimizeMolecule(mol)
    except:
        return False

    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        preparator = MoleculePreparation()
        setup = preparator.prepare(mol)[0]
        pdbqt_string, is_ok, err_msg = PDBQTWriterLegacy.write_string(setup)
        if is_ok:
            with open(output_path, 'w') as f:
                f.write(pdbqt_string)
            return True
        else:
            print(f"  meeko failed: {err_msg}")
            return False
    except Exception as e:
        print(f"  meeko error: {e}")
        return False


def get_pocket_center(pdb_path, chain_id='A'):
    """Get geometric center of the receptor as docking center."""
    if not HAS_BIO:
        return (0, 0, 0)
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('rec', pdb_path)
    except:
        return (0, 0, 0)
    chain = structure[0][chain_id]
    coords = []
    for residue in chain:
        for atom in residue:
            coords.append(atom.get_coord())
    center = np.mean(coords, axis=0)
    return tuple(center)


def run_vina(receptor_pdbqt, ligand_pdbqt, output_path, center, box_size=26):
    """Run AutoDock Vina docking."""
    cx, cy, cz = center
    cmd = [
        VINA_BIN,
        '--receptor', receptor_pdbqt,
        '--ligand', ligand_pdbqt,
        '--out', output_path,
        '--center_x', str(cx),
        '--center_y', str(cy),
        '--center_z', str(cz),
        '--size_x', str(box_size),
        '--size_y', str(box_size),
        '--size_z', str(box_size),
        '--exhaustiveness', '8',
        '--num_modes', '3',
        '--energy_range', '3',
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            err = result.stderr[:300]
            print(f"Vina failed: {err}")
            return None

        # Parse best score from OUTPUT FILE (Vina writes result as comment in PDBQT)
        best_score = None
        if os.path.exists(output_path):
            with open(output_path) as f:
                for line in f:
                    if line.startswith('REMARK VINA RESULT:'):
                        parts = line.split()
                        if len(parts) >= 4:
                            best_score = float(parts[3])
                            break

        # Fallback: try stdout if file parsing failed
        if best_score is None:
            for line in result.stdout.split('\n'):
                if 'REMARK VINA RESULT:' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        best_score = float(parts[3])
                        break

        return best_score
    except subprocess.TimeoutExpired:
        print(f"  Vina timeout")
        return None
    except Exception as e:
        print(f"  Vina error: {e}")
        return None


def dock_smiles(smiles, receptor_pdbqt, center, name="ligand", work_dir="/tmp"):
    """Full docking pipeline for one molecule. Returns (binding_energy, pdbqt_path)."""
    os.makedirs(work_dir, exist_ok=True)

    # Prepare ligand PDBQT
    lig_pdbqt = os.path.join(work_dir, f"{name.replace('/', '_').replace(' ', '_')}.pdbqt")
    ok = smiles_to_pdbqt(smiles, lig_pdbqt, name=name)
    if not ok:
        return None, None

    # Run Vina
    out_pdbqt = os.path.join(work_dir, f"{name.replace('/', '_').replace(' ', '_')}_out.pdbqt")
    score = run_vina(receptor_pdbqt, lig_pdbqt, out_pdbqt, center)
    return score, out_pdbqt


def batch_dock(molecules_smiles_list, receptor_pdb="pdb_cache/6lgx.pdb",
               chain_id='A', top_n=20, work_dir="/tmp/vina_dock", box_size=26):
    """Batch dock multiple molecules. molecules_smiles_list: list of (smiles, name)."""
    print(f"Batch docking {min(top_n, len(molecules_smiles_list))} molecules...")

    # Prepare receptor
    receptor_pdbqt = os.path.join(work_dir, "receptor.pdbqt")
    os.makedirs(work_dir, exist_ok=True)
    if not os.path.exists(receptor_pdbqt):
        print("  Preparing receptor PDBQT from", os.path.basename(receptor_pdb))
        ok = pdb_to_pdbqt(receptor_pdb, receptor_pdbqt, chain_id=chain_id)
        if not ok:
            print("  Failed to prepare receptor")
            return []
    else:
        print("  Using cached receptor PDBQT")

    # Get pocket center (center of whole protein for blind docking)
    center = get_pocket_center(receptor_pdb, chain_id=chain_id)
    print(f"  Docking center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
    print(f"  Box size: {box_size}A")

    # Dock each molecule
    results = []
    for i, (smiles, name) in enumerate(molecules_smiles_list[:top_n]):
        print(f"  Docking {i+1}/{min(top_n, len(molecules_smiles_list))}: {str(name)[:30]}...", end=" ", flush=True)
        score, out_path = dock_smiles(smiles, receptor_pdbqt, center,
                                       name=f"mol_{i}", work_dir=work_dir)
        if score is not None:
            results.append({'smiles': smiles, 'name': str(name), 'binding_energy': score})
            print(f" {score:.2f} kcal/mol")
        else:
            print(" FAILED")

    # Sort by binding energy (more negative = better)
    results.sort(key=lambda x: x['binding_energy'])

    print(f"\n  Top 5 by binding energy:")
    for i, r in enumerate(results[:5]):
        print(f"    {i+1}. {str(r['name'])[:30]:30} {r['binding_energy']:.2f} kcal/mol  {r['smiles'][:40]}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--receptor', default='pdb_cache/6lgx.pdb')
    parser.add_argument('--chain', default='A')
    parser.add_argument('--smiles', type=str, default=None)
    parser.add_argument('--batch', type=str, default=None, help='JSON file with molecules')
    parser.add_argument('--top-n', type=int, default=20)
    parser.add_argument('--work-dir', default='/tmp/vina_dock')
    parser.add_argument('--box-size', type=int, default=26)
    args = parser.parse_args()

    if args.smiles:
        results = batch_dock([(args.smiles, "custom")], args.receptor, args.chain,
                              args.top_n, args.work_dir, args.box_size)
        if results:
            print(f"\nBinding energy: {results[0]['binding_energy']:.2f} kcal/mol")
        else:
            print("Docking failed")

    elif args.batch:
        with open(args.batch) as f:
            data = json.load(f)
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                molecules = [(d['smiles'], d.get('name', 'unknown')) for d in data]
            else:
                molecules = [(d, f"mol_{i}") for i, d in enumerate(data)]
        else:
            print("Unknown JSON format")
            return

        results = batch_dock(molecules, args.receptor, args.chain,
                              args.top_n, args.work_dir, args.box_size)
        out_path = 'docking_results.json'
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {len(results)} results to {out_path}")

    else:
        # Default: dock known drugs
        drugs = [
            ('CC(=O)Oc1ccccc1C(=O)O', 'Aspirin'),
            ('CC(=O)NC1=CC=CC=C1', 'Acetaminophen'),
            ('CN(C)CCCC(c1ccc(F)cc1)c1ccccc1', 'Fluoxetine'),
            ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'Caffeine'),
            ('CC(C)Cc1ccc(C(C)C(=O)O)cc1', 'Ibuprofen'),
            ('OC(=O)C1=CC=CC=C1O', 'Salicylic acid'),
            ('CN(C)CC(CO)c1ccc(O)cc1', 'Salbutamol'),
        ]
        results = batch_dock(drugs, args.receptor, args.chain,
                              args.top_n, args.work_dir, args.box_size)
        with open('docking_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {len(results)} results to docking_results.json")


if __name__ == '__main__':
    main()
