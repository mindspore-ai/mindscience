#!/usr/bin/env python3
"""Chemistry Query chain entry point - standard agent interface."""
import argparse
import json
import sys
import os
import subprocess
from datetime import datetime, timezone
from rdkit import Chem

script_dir = os.path.dirname(os.path.abspath(__file__))

def parse_input():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input-json')
    args = parser.parse_args()
    if args.input_json:
        input_data = json.loads(args.input_json)
    else:
        input_data = json.load(sys.stdin)
    smiles = input_data.get('smiles')
    name = input_data.get('name')
    if not smiles:
        if not name:
            raise ValueError('Require "smiles" or "name" in input')
        proc = subprocess.run(
            [sys.executable, os.path.join(script_dir, 'query_pubchem.py'),
             '--compound', name, '--type', 'structure', '--format', 'smiles'],
            cwd=script_dir, capture_output=True, text=True, timeout=30)
        if proc.returncode:
            raise RuntimeError(f'PubChem lookup failed for "{name}": {proc.stderr.strip()}')
        smiles_str = proc.stdout.strip()
        if not smiles_str:
            raise RuntimeError(f'No SMILES from PubChem for "{name}"')
        smiles_data = json.loads(smiles_str)
        smiles = smiles_data.get('result')
        if not smiles:
            raise RuntimeError(f'No SMILES in PubChem response for "{name}"')
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError('Invalid SMILES: ' + smiles)
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return canonical_smiles, input_data

def main():
    canonical_smiles = ''
    try:
        canonical_smiles, input_data = parse_input()
        viz_dir = os.path.join(script_dir, '..', 'viz')
        os.makedirs(viz_dir, exist_ok=True)
        safe_name = input_data.get('name', 'unknown_mol')[:30].replace('/', '_').replace('\\', '_')
        png_path = os.path.join(viz_dir, safe_name + '.png')
        report = {}
        warnings = []

        # Props
        proc = subprocess.run(
            [sys.executable, os.path.join(script_dir, 'rdkit_mol.py'),
             '--smiles', canonical_smiles, '--action', 'props'],
            cwd=script_dir, capture_output=True, text=True, timeout=30)
        if proc.returncode == 0:
            report['props'] = json.loads(proc.stdout)
        else:
            warnings.append(proc.stderr.strip())

        # Draw
        proc = subprocess.run(
            [sys.executable, os.path.join(script_dir, 'rdkit_mol.py'),
             '--smiles', canonical_smiles, '--action', 'draw', '--output', png_path],
            cwd=script_dir, capture_output=True, text=True, timeout=30)
        if proc.returncode == 0:
            report['draw'] = json.loads(proc.stdout)
        else:
            warnings.append(proc.stderr.strip())

        viz_files = [png_path] if os.path.exists(png_path) else []

        # Retro
        proc = subprocess.run(
            [sys.executable, os.path.join(script_dir, 'rdkit_mol.py'),
             '--target', canonical_smiles, '--action', 'retro', '--depth', '2'],
            cwd=script_dir, capture_output=True, text=True, timeout=30)
        if proc.returncode == 0:
            report['retro'] = json.loads(proc.stdout)
        else:
            warnings.append(proc.stderr.strip())

        status = 'success'
        confidence = 0.95
        risks = []
    except Exception as e:
        status = 'error'
        report = {}
        confidence = 0.0
        warnings = [str(e)]
        risks = []
        viz_files = []

    print(json.dumps({
        'agent': 'chemistry-query',
        'version': '1.4.0',
        'smiles': canonical_smiles,
        'status': status,
        'report': report,
        'risks': risks,
        'viz': viz_files,
        'recommend_next': ['pharmacology', 'toxicology'],
        'confidence': confidence,
        'warnings': warnings,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }, indent=2, default=str))

if __name__ == '__main__':
    main()
