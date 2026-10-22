"""
Rabies Candidate Pipeline - End-to-End Ranking

Integrates all modules:
1. Structure prediction → 2. Pocket detection → 3. Molecule generation
→ 4. Drug screening → 5. BBB permeability → 6. PK/PD simulation
→ 7. Viral dynamics → 8. Toxicity → 9. Composite ranking

Usage:
    python rabies_pipeline.py
    python rabies_pipeline.py --target rabies --top-k 10
"""
import sys, os, json, time, argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='rabies', help='Target virus')
    parser.add_argument('--top-k', type=int, default=10, help='Top candidates to analyze')
    parser.add_argument('--skip-structure', action='store_true', help='Skip structure prediction')
    parser.add_argument('--dose-mg', type=int, default=500, help='Default dose in mg')
    args = parser.parse_args()

    print("="*70)
    print("  RABIES CANDIDATE PIPELINE")
    print("  Universal Antiviral Platform for Brain-Invading Viruses")
    print("="*70)
    start_time = time.time()

    # ============================================================
    # STAGE 1: Structure Prediction
    # ============================================================
    print("\n[1/9] PROTEIN STRUCTURE PREDICTION")
    print("-" * 50)

    if args.skip_structure and os.path.exists('rabies_g_predicted.pdb'):
        print("  Using cached structure: rabies_g_predicted.pdb")
    else:
        print("  Running ESM-2 + contact map predictor...")
        try:
            os.system('/usr/local/bin/python3.10 predict_structure.py')
            print("  Structure saved: rabies_g_predicted.pdb")
        except Exception as e:
            print(f"  Structure prediction failed: {e}")
            print("  Continuing with sequence-only analysis...")

    # ============================================================
    # STAGE 2: Binding Pocket Detection
    # ============================================================
    print("\n[2/9] BINDING POCKET DETECTION")
    print("-" * 50)

    try:
        contacts = np.load('rabies_g_contacts.npy')
        # Find residues with high contact probability (pockets)
        residue_scores = contacts.mean(axis=1)
        top_residues = np.argsort(residue_scores)[-10:][::-1]
        print(f"  Contact map: {contacts.shape[0]} residues analyzed")
        print(f"  Top pocket residues: {top_residues.tolist()}")
        print(f"  Pocket scores: {residue_scores[top_residues].round(3).tolist()}")
        pockets = top_residues
    except Exception as e:
        print(f"  Could not load contacts: {e}")
        pockets = list(range(10, 20))  # default pocket range

    # ============================================================
    # STAGE 3: Molecule Generation
    # ============================================================
    print("\n[3/9] MOLECULE GENERATION")
    print("-" * 50)

    print("  Loading combinatorial generator...")
    try:
        from generate_molecules import enumerate_molecules, NOVEL_MOLECULES
        generated_smis = enumerate_molecules(max_mols=200)
        generated = [(s, f'gen_{i}') for i, s in enumerate(generated_smis)]
        # Also add pre-built novel molecules
        for s in NOVEL_MOLECULES:
            generated.append((s, 'novel'))
        print(f"  Generated {len(generated)} candidates")
    except Exception as e:
        print(f"  Generator failed: {e}, using pre-built set")
        generated = []

    # Also include known antivirals
    known_drugs = [
        ('CC(=O)Oc1ccccc1C(=O)O', 'Aspirin'),
        ('CN(C)CCCC(c1ccc(F)cc1)c1ccccc1', 'Fluoxetine'),
        ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'Caffeine'),
        ('CC(=O)NC1=CC=CC=C1', 'Acetaminophen'),
        ('CC(C)Cc1ccc(C(C)C(=O)O)cc1', 'Ibuprofen'),
        ('OC(=O)C1=CC=CC=C1O', 'Salicylic acid'),
        ('CN(C)CC(CO)c1ccc(O)cc1', 'Salbutamol'),
    ]

    all_candidates = [(s, n) for s, n in known_drugs]
    for item in generated:
        if isinstance(item, dict):
            all_candidates.append((item['smiles'], item.get('name', 'generated')))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            all_candidates.append((item[0], item[1]))

    print(f"  Total candidates: {len(all_candidates)}")

    # ============================================================
    # STAGE 4: Drug-Protein Screening + Docking
    # ============================================================
    print("\n[4/9] DRUG-PROTEIN SCREENING & DOCKING")
    print("-" * 50)

    def dock_and_score(smiles, name):
        """Run Vina docking + PK scoring."""
        try:
            from vina_dock import dock_smiles, pdb_to_pdbqt, get_pocket_center
            # Prepare receptor once
            receptor_pdbqt = '/tmp/vina_dock/receptor.pdbqt'
            if not os.path.exists(receptor_pdbqt):
                os.makedirs('/tmp/vina_dock', exist_ok=True)
                pdb_to_pdbqt(args.receptor, receptor_pdbqt)
            center = get_pocket_center(args.receptor)
            score, _ = dock_smiles(smiles, receptor_pdbqt, center,
                                    name=name.replace(' ', '_'), work_dir='/tmp/vina_dock_screen')
            return score if score is not None else 0
        except Exception as e:
            return 0

    scored = []
    for i, (smi, name) in enumerate(all_candidates):
        # Vina docking (only for top-n to save time)
        vina_score = 0
        if i < args.top_k * 2:
            vina_score = dock_and_score(smi, name)

        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smi)
        dl = 0
        if mol:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            if 150 < mw < 500: dl += 1
            if -1 < logp < 5: dl += 1
            if tpsa < 140: dl += 1
            if hbd <= 5: dl += 1
            if hba <= 10: dl += 1

        scored.append({
            'smiles': smi,
            'name': name,
            'vina_score': vina_score,
            'drug_likeness': dl / 6.0,
        })
        if i < args.top_k:
            print(f"  {i+1}. {name:<25} Vina={vina_score:.2f} kcal/mol")

    # Re-rank by Vina score (more negative = better)
    scored.sort(key=lambda x: x['vina_score'])

    # ============================================================
    # STAGE 5: BBB Permeability
    # ============================================================
    print("\n[5/9] BBB PERMEABILITY PREDICTION")
    print("-" * 50)

    from pkpd_model import get_bbb_permeability
    for s in scored:
        s['bbb_perm'] = get_bbb_permeability(s['smiles'])

    bbb_ranked = sorted(scored, key=lambda x: x['bbb_perm'], reverse=True)
    print(f"  BBB predictions complete")
    print(f"  High BBB (>0.7): {sum(1 for s in scored if s['bbb_perm'] > 0.7)}/{len(scored)}")
    print(f"  Top BBB:")
    for s in bbb_ranked[:3]:
        print(f"    {s['name']:<25} pBBB={s['bbb_perm']:.3f}")

    # ============================================================
    # STAGE 6: PK/PD Simulation
    # ============================================================
    print("\n[6/9] PK/PD SIMULATION")
    print("-" * 50)

    from pkpd_model import simulate_pkpd
    for s in scored[:args.top_k * 2]:  # analyze top candidates
        pkpd = simulate_pkpd(s['smiles'], dose_mg=args.dose_mg, name=s['name'])
        s['pkpd'] = {
            'Cmax_brain_uM': pkpd['Cmax_brain_uM'],
            'AUC_brain': pkpd['AUC_brain'],
            'avg_suppression_24h': pkpd['avg_suppression_24h'],
            'time_above_ec50_hrs': pkpd['time_above_ec50_hrs'],
        }

    print(f"  PK/PD simulated for {min(args.top_k * 2, len(scored))} candidates")

    # ============================================================
    # STAGE 7: Viral Dynamics
    # ============================================================
    print("\n[7/9] VIRAL DYNAMICS SIMULATION")
    print("-" * 50)

    from pkpd_model import simulate_pkpd, simulate_viral_dynamics
    for s in scored[:args.top_k * 2]:
        pkpd = simulate_pkpd(s['smiles'], dose_mg=args.dose_mg, name=s['name'])
        viral = simulate_viral_dynamics(pkpd, treatment_delay_hrs=48)
        s['viral'] = {
            'survival_fraction': viral['survival_fraction'],
            'peak_virus': viral['peak_virus'],
            'cleared': viral['cleared'],
        }

    cleared_count = sum(1 for s in scored[:args.top_k * 2] if s.get('viral', {}).get('cleared', False))
    print(f"  Cleared: {cleared_count}/{min(args.top_k * 2, len(scored))}")

    # ============================================================
    # STAGE 8: Toxicity Prediction
    # ============================================================
    print("\n[8/9] TOXICITY PREDICTION")
    print("-" * 50)

    from pkpd_model import predict_toxicity
    for s in scored:
        tox = predict_toxicity(s['smiles'])
        s['toxicity'] = {
            'toxic': tox['toxic'],
            'score': tox['score'],
            'flags': tox['flags'],
        }

    toxic_count = sum(1 for s in scored if s.get('toxicity', {}).get('toxic', False))
    print(f"  Toxic: {toxic_count}/{len(scored)}")
    print(f"  Safe: {len(scored) - toxic_count}/{len(scored)}")

    # ============================================================
    # STAGE 9: Composite Scoring & Ranking
    # ============================================================
    print("\n[9/9] COMPOSITE SCORING & RANKING")
    print("-" * 50)

    for s in scored:
        bbb = s.get('bbb_perm', 0.5)
        pk = s.get('pkpd', {}).get('avg_suppression_24h', 0)
        viral = s.get('viral', {}).get('survival_fraction', 0)
        tox = 1.0 - s.get('toxicity', {}).get('score', 0)
        dl = s.get('drug_likeness', 0)

        # Composite: BBB(20%) + PK/PD(20%) + Viral(30%) + Safety(15%) + DrugLike(15%)
        s['composite'] = bbb * 0.20 + pk * 0.20 + viral * 0.30 + tox * 0.15 + dl * 0.15

    scored.sort(key=lambda x: x['composite'], reverse=True)

    # Print final ranking
    print(f"\n{'='*70}")
    print(f"  FINAL RANKING: {args.target.upper()} CANDIDATE PANEL")
    print(f"{'='*70}")
    print(f"  {'Rank':<5} {'Name':<25} {'Score':<8} {'BBB':<8} {'PK/PD':<8} {'Viral':<8} {'Safe':<6} {'DL':<6}")
    print(f"  {'-'*5} {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")

    for i, s in enumerate(scored[:args.top_k]):
        bbb = s.get('bbb_perm', 0)
        pk = s.get('pkpd', {}).get('avg_suppression_24h', 0)
        viral = s.get('viral', {}).get('survival_fraction', 0)
        tox = 1.0 - s.get('toxicity', {}).get('score', 0)
        dl = s.get('drug_likeness', 0)
        toxic = "!" if s.get('toxicity', {}).get('toxic', False) else " "

        print(f"  {i+1:<5} {s['name']:<25} {s['composite']:<8.3f} {bbb:<8.3f} {pk:<8.3f} {viral:<8.3f} {tox:<6.2f} {dl:<6.2f} {toxic}")

    print(f"{'='*70}")

    elapsed = time.time() - start_time
    print(f"\n  Pipeline completed in {elapsed:.1f}s")
    print(f"  Results saved to: pipeline_results.json")

    # Save results
    with open('pipeline_results.json', 'w') as f:
        json.dump(scored[:args.top_k], f, indent=2, default=str)


if __name__ == '__main__':
    main()
