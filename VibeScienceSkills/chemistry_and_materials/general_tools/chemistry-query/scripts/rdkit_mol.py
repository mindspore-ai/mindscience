import argparse
import json
import sys
import re
import os
import subprocess
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Descriptors, Draw, AllChem
from rdkit.Chem import BRICS
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdChemReactions

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol

def resolve_target(target):
    """Resolve name to SMILES using PubChem"""
    mol_try = Chem.MolFromSmiles(target)
    if mol_try is not None:
        return target
    # PubChem lookup
    script_dir = os.path.dirname(__file__)
    proc = subprocess.run([sys.executable, "query_pubchem.py", "--compound", target, "--type", "structure", "--format", "smiles"],
                          cwd=script_dir, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ValueError(f"PubChem lookup failed for '{target}': {proc.stderr.strip()}")
    out = proc.stdout.strip()
    if not out:
        raise ValueError(f"No SMILES found for '{target}'")
    data = json.loads(out)
    return data["result"]

def brics_retro(mol, depth):
    """Recursive BRICS retrosynthesis. Returns list of fragment SMILES."""
    if depth <= 0:
        return [Chem.MolToSmiles(mol, isomericSmiles=True)]
    try:
        frags = BRICS.BRICSDecompose(mol)  # returns set of SMILES strings
        if not frags or len(frags) <= 1:
            return [Chem.MolToSmiles(mol, isomericSmiles=True)]
        precursors = []
        for frag_smi in frags:
            # Remove BRICS dummy atom labels [n*] for cleaner output
            clean_smi = re.sub(r'\[\d+\*\]', '[H]', frag_smi)
            frag_mol = Chem.MolFromSmiles(clean_smi)
            if frag_mol is not None:
                precursors += brics_retro(frag_mol, depth - 1)
            else:
                precursors.append(frag_smi)
        return list(set(precursors))
    except Exception:
        return [Chem.MolToSmiles(mol, isomericSmiles=True)]

def main():
    parser = argparse.ArgumentParser(description="RDKit molecule processing")
    parser.add_argument("--smiles", help="Single SMILES for props/draw/fingerprint/xyz")
    parser.add_argument("--query_smiles", help="Query SMILES for similarity/substruct")
    parser.add_argument("--target_smiles", help="Comma-separated target SMILES for similarity/substruct")
    parser.add_argument("--reactants", help="Comma/space separated reactant SMILES for react")
    parser.add_argument("--smarts", help="SMARTS reaction template for react")
    parser.add_argument("--target", help="Target SMILES/name for retro/plan")
    parser.add_argument("--depth", type=int, default=1, help="Retro depth")
    parser.add_argument("--steps", type=int, default=3, help="Plan steps")
    parser.add_argument("--templates", help="Comma-separated template names filter")
    parser.add_argument("--action", choices=["props", "draw", "fingerprint", "similarity", "substruct", "xyz", "react", "retro", "plan"], default="props", help="Action")
    parser.add_argument("--output", help="Output file for draw")
    parser.add_argument("--format", choices=["png", "svg"], default="png", help="Draw format")
    parser.add_argument("--radius", type=int, default=2, help="Morgan radius")
    args = parser.parse_args()

    try:
        if args.action == "props":
            mol = get_mol(args.smiles)
            props = {
                "mw": Descriptors.ExactMolWt(mol),
                "logp": Descriptors.MolLogP(mol),
                "tpsa": Descriptors.TPSA(mol),
                "hbd": Descriptors.NumHDonors(mol),
                "hba": Descriptors.NumHAcceptors(mol),
                "rotb": Descriptors.NumRotatableBonds(mol),
                "arom_rings": Descriptors.NumAromaticRings(mol),
            }
            print(json.dumps(props, indent=2))

        elif args.action == "fingerprint":
            mol = get_mol(args.smiles)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, args.radius, nBits=2048)
            print(json.dumps({
                "bitstring": fp.ToBitString(),
                "num_bits": fp.GetNumBits(),
                "bits_set": sorted(list(fp.GetOnBits()))
            }, indent=2))

        elif args.action in ["similarity", "substruct"]:
            if not args.query_smiles or not args.target_smiles:
                raise ValueError("Need both --query_smiles and --target_smiles")
            qmol = get_mol(args.query_smiles)
            target_smiles_list = [s.strip() for s in args.target_smiles.split(',') if s.strip()]
            target_mols = [get_mol(s) for s in target_smiles_list]
            if args.action == "similarity":
                qfp = AllChem.GetMorganFingerprintAsBitVect(qmol, args.radius, nBits=2048)
                sims = [DataStructs.TanimotoSimilarity(qfp, AllChem.GetMorganFingerprintAsBitVect(t, args.radius, nBits=2048)) for t in target_mols]
                results = [{"target": target_smiles_list[i], "value": sims[i]} for i in range(len(sims))]
                print(json.dumps({"results": results, "max_value": max(sims) if sims else 0}, indent=2))
            else:  # substruct
                matches = [t.HasSubstructMatch(qmol) for t in target_mols]
                results = [{"target": target_smiles_list[i], "match": matches[i]} for i in range(len(matches))]
                print(json.dumps({"results": results, "num_hits": sum(matches)}, indent=2))

        elif args.action == "draw":
            mol = get_mol(args.smiles)
            output = args.output or f"mol.{args.format}"
            if args.format == "svg":
                drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
                drawer.DrawMolecule(mol)
                drawer.FinishDrawing()
                svg = drawer.GetDrawingText()
                print(json.dumps({"svg": svg, "success": True}, indent=2))
                with open(output, "w") as f:
                    f.write(svg)
            else:  # png
                from PIL import Image
                img = Draw.MolToImage(mol, size=(300,300))
                img.save(output)
                print(json.dumps({"image_path": output, "success": True}, indent=2))

        elif args.action == "xyz":
            mol = Chem.MolFromSmiles(args.smiles)
            if mol is None:
                raise ValueError("Invalid SMILES")
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            xyz = Chem.MolToXYZBlock(mol)
            print(json.dumps({"xyz": xyz.strip(), "num_atoms": mol.GetNumAtoms()}, indent=2))

        elif args.action == "react":
            if not args.reactants or not args.smarts:
                raise ValueError("Need both --reactants and --smarts")
            reactant_smiles = re.split(r'[,; \\t\\n]+', args.reactants)
            reactant_smiles = [s.strip() for s in reactant_smiles if s.strip()]
            reactants_mols = [get_mol(smi) for smi in reactant_smiles]
            rxn = rdChemReactions.ReactionFromSmarts(args.smarts)
            if rxn is None:
                raise ValueError("Invalid SMARTS reaction template")
            products = rxn.RunReactants(reactants_mols)
            product_smiles = []
            for prod_tuple in products:
                for mol in prod_tuple:
                    if mol is not None:
                        try:
                            Chem.SanitizeMol(mol)
                            if mol.GetNumAtoms() > 0:
                                product_smiles.append(Chem.MolToSmiles(mol, isomericSmiles=True))
                        except:
                            pass  # skip invalid
            print(json.dumps({"products": product_smiles, "num_products": len(product_smiles)}, indent=2))

        elif args.action == "retro":
            target_smiles = resolve_target(args.target or args.smiles)
            mol = get_mol(target_smiles)
            precursors = brics_retro(mol, args.depth)
            print(json.dumps({
                "action": "retro",
                "target": target_smiles,
                "depth": args.depth,
                "precursors": sorted(precursors),
                "num_precursors": len(precursors)
            }, indent=2))

        elif args.action == "plan":
            target_smiles = resolve_target(args.target or args.smiles)
            mol = get_mol(target_smiles)
            # Simple multi-step BRICS retro plan (BFS-like, limited)
            current_products = [target_smiles]
            route = []
            for step_num in range(1, args.steps + 1):
                precursors_step = []
                for prod_smiles in current_products:
                    prod_mol = get_mol(prod_smiles)
                    step_precs = brics_retro(prod_mol, 1)
                    precursors_step.extend(step_precs)
                precursors_step = list(set(precursors_step))
                route.append({
                    "step": step_num,
                    "precursors": precursors_step[:10],  # limit
                    "cond": "BRICS template (ester/amide/etc disconnects)",
                    "yield": "N/A (BRICS)",
                    "product": current_products[0]  # main
                })
                current_products = precursors_step[:5]  # branch factor limit
            print(json.dumps({
                "action": "plan",
                "target": target_smiles,
                "steps": args.steps,
                "templates_used": "BRICS",
                "route": route
            }, indent=2))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()