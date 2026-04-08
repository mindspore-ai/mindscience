# PubChem API Endpoints

This reference provides key endpoints for the PubChem REST API. For full documentation, visit https://pubchemdocs.ncbi.nlm.nih.gov/pug-rest.

## Base URLs
- PUG REST: https://pubchem.ncbi.nlm.nih.gov/rest/pug
- PUG View: https://pubchem.ncbi.nlm.nih.gov/rest/pug_view

## Common Endpoints

### Get Compound ID (CID)
- GET /compound/name/{compound_name}/cids/JSON
- Example: /compound/name/aspirin/cids/JSON

### Compound Properties
- GET /compound/cid/{cid}/property/{property_list}/JSON
- Properties: MolecularFormula, MolecularWeight, IUPACName, InChIKey, etc.
- Example: /compound/cid/2244/property/MolecularFormula,MolecularWeight/JSON

### Structure Data
- SMILES: /compound/cid/{cid}/smiles/TXT
- InChI: /compound/cid/{cid}/inchi/TXT
- Image: /compound/cid/{cid}/PNG?image_size=large

### Synthesis Information
- GET /data/compound/{cid}/JSON?heading=Synthesis (via PUG View)
- Returns sections related to synthesis if available, including references and methods.

### Literature and Patents
- GET /compound/cid/{cid}/literature/JSON

## Rate Limits
- PubChem allows up to 5 requests per second, 400 requests per minute.
- Handle errors and retries in scripts.

## Examples
- Retrieve aspirin info: requests.get('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin/property/MolecularWeight/JSON')
