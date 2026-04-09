import argparse
import requests
import json
import sys

def main():
    parser = argparse.ArgumentParser(description="ChEMBL compound bioactivity")
    parser.add_argument("--compound", required=True, help="Compound name/InChIKey/SMILES")
    parser.add_argument("--target", help="Specific target name (filter)")
    parser.add_argument("--type", choices=["assay", "mechanism"], default="assay")
    args = parser.parse_args()

    base_url = "https://www.ebi.ac.uk/chembl/api/data"

    # Search molecule
    search_url = f"{base_url}/molecule/search"
    params = {"q": args.compound, "limit": 1}
    resp = requests.get(search_url, params=params)
    if resp.status_code != 200:
        print(json.dumps({"error": "Search failed", "status": resp.status_code}))
        sys.exit(1)
    data = resp.json()
    molecules = data.get("molecules", [])
    if not molecules:
        print(json.dumps({"error": "Compound not found"}))
        sys.exit(1)
    mol_chembl = molecules[0]["molecule_chembl_id"]

    if args.type == "assay":
        assay_url = f"{base_url}/molecule/{mol_chembl}/assay_count"
        assays = requests.get(assay_url).json()
        print(json.dumps({"chembl_id": mol_chembl, "assay_count": assays.get("assay_count", 0)}))
        
        # Top assays
        activity_url = f"{base_url}/molecule/{mol_chembl}/activities"
        params = {"limit": 10, "order_by": "-standard_value"}
        resp = requests.get(activity_url, params=params)
        activities = resp.json().get("activities", [])
        formatted = [{"assay": a["assay_chembl_id"], "target": a.get("target_chembl_id"), "type": a["activity_type"], "std_value": a.get("standard_value"), "std_unit": a.get("standard_units")} for a in activities]
        print(json.dumps({"top_assays": formatted}))
    elif args.type == "mechanism":
        mech_url = f"{base_url}/mechanism/{mol_chembl}"
        resp = requests.get(mech_url)
        mechanisms = resp.json().get("mechanisms", [])
        formatted = [{"target": m["target_chembl_id"], "mechanism": m["mechanism_of_action"], "action_type": m["action_type"]} for m in mechanisms]
        print(json.dumps({"mechanisms": formatted}))

if __name__ == "__main__":
    main()