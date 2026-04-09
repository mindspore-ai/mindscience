import argparse
import json
import sys
from rdkit import Chem
from rdkit.Chem import Descriptors

def lipinski(mol):
    hba = Descriptors.NumHAcceptors(mol)
    hbd = Descriptors.NumHDonors(mol)
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    violations = sum([hba > 10, hbd > 5, mw > 500, logp > 5])
    return {"hba": hba, "hbd": hbd, "mw": mw, "logp": logp, "ro5_violations": violations}

def veber(mol):
    rotb = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    violations = sum([rotb > 10, tpsa > 140])
    return {"rotb": rotb, "tpsa": tpsa, "veber_violations": violations}

def alerts(mol):
    smiles = Chem.MolToSmiles(mol)
    pains = ['c1ncnc2c1ncn2', 'cc1ccc2nsnc2c1']  # example PAINS
    alerts = [p for p in pains if p in smiles]
    return {"pains_alerts": alerts, "num_alerts": len(alerts)}

def main():
    parser = argparse.ArgumentParser(description="RDKit ADMET rule-based")
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--target_smiles", help="Batch comma list")
    args = parser.parse_args()

    def process(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"smiles": smiles, "error": "Invalid"}
        return {"smiles": smiles, **lipinski(mol), **veber(mol), **alerts(mol)}

    if args.target_smiles:
        results = [process(s.strip()) for s in args.target_smiles.split(",")]
        print(json.dumps({"screen": results}))
    else:
        result = process(args.smiles)
        print(json.dumps(result))

if __name__ == "__main__":
    main()