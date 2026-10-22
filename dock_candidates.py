"""
dock_candidates.py — Dock DILIrank low-toxicity candidates against 6LGX.
Loads saved predictions, strips salts, docks all.
"""
import sys, os, json, subprocess, numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

VINA_BIN = "/usr/local/bin/vina"
DATA_DIR = 'data'
WORK_DIR = '/tmp/repurpose_screen'
os.makedirs(WORK_DIR, exist_ok=True)

# Load predictions
with open(os.path.join(DATA_DIR, 'repurposing_results.json')) as f:
    data = json.load(f)

# Filter: DILI class 0 or 1, <4 Tox21 flags
candidates = [r for r in data['all_predictions']
              if r['dil_class'] in (0, 1) and r['tx_flags'] < 4]
candidates.sort(key=lambda r: (r['dil_class'], r['tx_flags'], r['ct_tox']))
print(f'Candidates: {len(candidates)}')

# Strip salts: keep largest organic fragment
def strip_salt(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) == 1:
        return smiles
    # Keep the largest fragment by heavy atom count
    frags_sorted = sorted(frags, key=lambda m: m.GetNumHeavyAtoms(), reverse=True)
    return Chem.MolToSmiles(frags_sorted[0])

def smiles_to_pdbqt(smiles, output_path):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
    except:
        return False
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        setup = MoleculePreparation().prepare(mol)[0]
        pdbqt_string, is_ok, err_msg = PDBQTWriterLegacy.write_string(setup)
        if is_ok:
            with open(output_path, 'w') as f:
                f.write(pdbqt_string)
            return True
        return False
    except:
        return False

# Prepare receptor
rec_pdbqt = os.path.join(WORK_DIR, 'receptor.pdbqt')
if not os.path.exists(rec_pdbqt):
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    rec_pdb = os.path.join(os.path.dirname(DATA_DIR), 'pdb_cache/6lgx.pdb')
    structure = parser.get_structure('rec', rec_pdb)
    chain = structure[0]['A']
    ATOM_TYPE_TABLE = {6: 'C', 7: 'N', 8: 'O', 16: 'S', 9: 'F', 17: 'Cl', 35: 'Br', 53: 'I', 15: 'P'}
    AA_CODES = {'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS',
                'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR'}
    with open(rec_pdbqt, 'w') as f:
        f.write("REMARK PDBQT\n")
        serial = 0
        for residue in chain:
            if residue.get_resname().strip() not in AA_CODES:
                continue
            for atom in residue:
                serial += 1
                if atom.get_full_id()[3][0].strip():
                    continue
                elem = atom.element.strip() if atom.element else atom.get_name()[0]
                if not elem:
                    continue
                x, y, z = atom.get_coord()
                elem_num = {'H':1,'C':6,'N':7,'O':8,'S':16,'F':9,'CL':17,'BR':35,'I':53,'P':15}.get(elem.upper(), 6)
                atype = ATOM_TYPE_TABLE.get(elem_num, 'C')
                aname = f" {atom.get_name().strip():<3}"
                line = f"ATOM  {serial:>5} {aname:<4} {residue.get_resname():<3} A{residue.get_id()[1]:>4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atype}\n"
                f.write(line)
        f.write("TER\nEND\n")

# Center
from Bio.PDB import PDBParser
parser = PDBParser(QUIET=True)
rec_pdb = os.path.join(os.path.dirname(DATA_DIR), 'pdb_cache/6lgx.pdb')
structure = parser.get_structure('rec', rec_pdb)
coords = [atom.get_coord() for residue in structure[0]['A'] for atom in residue]
center = tuple(np.mean(coords, axis=0))

# Dock
docking_results = []
for i, r in enumerate(candidates):
    name = r['drug'].replace('/', '_').replace(' ', '_')[:40]
    smiles = strip_salt(r['smiles'])
    if smiles is None:
        print(f'  {i+1}/{len(candidates)}: {r["drug"][:25]:25} - salt strip FAIL, trying original')
        smiles = r['smiles']

    lig_pdbqt = os.path.join(WORK_DIR, f'{name}.pdbqt')
    out_pdbqt = os.path.join(WORK_DIR, f'{name}_out.pdbqt')

    # Skip if already docked
    if os.path.exists(out_pdbqt):
        with open(out_pdbqt) as f:
            score = None
            for line in f:
                if line.startswith('REMARK VINA RESULT:'):
                    score = float(line.split()[3])
                    break
        if score is not None:
            docking_results.append({**r, 'smiles_freebase': smiles, 'binding_energy': score})
            print(f'  {i+1}/{len(candidates)}: {r["drug"][:25]:25} - CACHED {score:.2f}')
            continue

    print(f'  {i+1}/{len(candidates)}: {r["drug"][:25]:25}... ', end='', flush=True)
    ok = smiles_to_pdbqt(smiles, lig_pdbqt)
    if not ok:
        print('FAIL (ligand prep)')
        continue

    cmd = [VINA_BIN, '--receptor', rec_pdbqt, '--ligand', lig_pdbqt,
           '--out', out_pdbqt,
           '--center_x', f'{center[0]:.1f}', '--center_y', f'{center[1]:.1f}',
           '--center_z', f'{center[2]:.1f}',
           '--size_x', '26', '--size_y', '26', '--size_z', '26',
           '--exhaustiveness', '8', '--num_modes', '1']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        score = None
        if os.path.exists(out_pdbqt):
            with open(out_pdbqt) as f:
                for line in f:
                    if line.startswith('REMARK VINA RESULT:'):
                        score = float(line.split()[3])
                        break
        if score is not None:
            docking_results.append({**r, 'smiles_freebase': smiles, 'binding_energy': score})
            print(f'{score:.2f}')
        else:
            print(f'FAIL ({result.stderr[:60]})')
    except subprocess.TimeoutExpired:
        print('TIMEOUT')

docking_results.sort(key=lambda x: x['binding_energy'])

print('\n' + '='*60)
print('DOCKING RESULTS')
print('='*60)
print(f'{"Rank":<5} {"Drug":<28} {"DILI":<10} {"Binding":<8} {"Flags":<5}')
print('-'*60)
for rank, r in enumerate(docking_results, 1):
    dil_name = ['Ambiguous','No-concern','Less-concern','Most-concern'][r['dil_class']]
    print(f'{rank:<5} {r["drug"]:<28} {dil_name:<10} {r["binding_energy"]:<8.2f} {r["tx_flags"]:<5}')

# Save
with open(os.path.join(DATA_DIR, 'docking_candidates.json'), 'w') as f:
    json.dump(docking_results, f, indent=2, default=str)
print(f'\nSaved {len(docking_results)} docking results to data/docking_candidates.json')
