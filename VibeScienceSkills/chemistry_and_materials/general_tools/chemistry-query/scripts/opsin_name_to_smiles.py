import argparse
import json
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="OPSIN IUPAC name to SMILES")
    parser.add_argument("--name", required=True, help="IUPAC/systematic name")
    parser.add_argument("--output", help="Output SMILES file")
    args = parser.parse_args()

    jar_path = os.path.join(os.path.dirname(__file__), "opsin.jar")
    if not os.path.exists(jar_path):
        print(json.dumps({"error": "opsin.jar missing—wget https://github.com/dan2097/opsin/releases/download/v2.8.0/opsin-core-2.8.0.jar"}), file=sys.stderr)
        sys.exit(1)

    cmd = ["java", "-jar", jar_path, "--stdin", "--output", "smiles"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate(input=args.name)
    if proc.returncode != 0:
        print(json.dumps({"error": stderr.strip()}), file=sys.stderr)
        sys.exit(1)
    
    smiles = stdout.strip()
    result = {"name": args.name, "smiles": smiles}
    print(json.dumps(result))
    if args.output:
        with open(args.output, 'w') as f:
            f.write(smiles)

if __name__ == "__main__":
    main()