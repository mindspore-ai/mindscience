import argparse
import json
import sys
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdChemReactions import ChemicalReaction

REACTION_TEMPLATES = {
    "amide": "[C:1](=O)[Cl:2].[N:3][H]>>[C:1](=O)[N:3]",  # Acid chloride + amine -> amide
    "ester": "[C:1](=O)[OH].[O:2][C:3]>>[C:1](=O)[O:2][C:3]",  # Acid + alcohol -> ester (no cat)
    "suzuki": "[c:1]{Br:2}.[B:3]([c:4])([c:5])[c:6].[Pd]>>[c:1][c:4]",  # Simplified Suzuki
}

def main():
    parser = argparse.ArgumentParser(description="RDKit reaction prediction/applier")
    parser.add_argument("--type", choices=["forward", "retro"], default="forward")
    parser.add_argument("--template", required=True, choices=list(REACTION_TEMPLATES.keys()))
    parser.add_argument("--reactants", nargs="+", help="SMILES reactants")
    parser.add_argument("--product", help="Target product for retro")
    args = parser.parse_args()

    if args.type == "forward" and not args.reactants:
        print(json.dumps({"error": "--reactants required"}), file=sys.stderr)
        sys.exit(1)
    if args.type == "retro" and not args.product:
        print(json.dumps({"error": "--product required"}), file=sys.stderr)
        sys.exit(1)

    rxn_smarts = REACTION_TEMPLATES[args.template]
    rxn = ChemicalReaction.FromSmarts(rxn_smarts)

    if args.type == "forward":
        mols = []
        for r in args.reactants:
            mol = Chem.MolFromSmiles(r.strip())
            if mol is None:
                print(json.dumps({"error": f"Invalid SMILES: {r}"}), file=sys.stderr)
                sys.exit(1)
            mols.append(mol)
        products = list(rxn.RunReactants(tuple(mols)))
        all_products = []
        for ps in products:
            for p in ps:
                Chem.SanitizeMol(p)
                all_products.append(Chem.MolToSmiles(p))
        print(json.dumps({"template": args.template, "reactants": args.reactants, "products": list(set(all_products))}))
    else:
        print(json.dumps({"warning": "Retro single-step placeholder; full planning needs tree search", "template": args.template, "product": args.product}))

if __name__ == "__main__":
    main()