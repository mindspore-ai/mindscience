import argparse
import requests
import json
import sys
from xml.etree import ElementTree as ET

def main():
    parser = argparse.ArgumentParser(description="PubMed literature search")
    parser.add_argument("--term", required=True, help="Search term e.g. 'aspirin synthesis'")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    esearch = f"{base_url}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": args.term.replace(" ", "+"),
        "retmax": args.limit,
        "retmode": "json"
    }
    resp = requests.get(esearch, params=params)
    if resp.status_code != 200:
        print(json.dumps({"error": "PubMed search failed"}))
        sys.exit(1)
    
    data = resp.json()
    ids = data["esearchresult"]["idlist"]
    
    if not ids:
        print(json.dumps({"term": args.term, "hits": 0}))
        sys.exit(0)
    
    # Fetch summaries
    efetch = f"{base_url}/efetch.fcgi"
    params = {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}
    resp = requests.get(efetch, params=params)
    root = ET.fromstring(resp.text)
    
    papers = []
    for pubmed in root.findall(".//PubmedArticle"):
        title = pubmed.find(".//ArticleTitle")
        authors = pubmed.find(".//AuthorList")
        pmid = pubmed.find(".//PMID")
        papers.append({
            "pmid": pmid.text if pmid is not None else "N/A",
            "title": title.text if title is not None else "N/A",
            "authors": [a.find("LastName").text for a in authors.findall("Author") if a.find("LastName") is not None][:3]
        })
    
    print(json.dumps({"term": args.term, "hits": len(ids), "papers": papers}))

if __name__ == "__main__":
    main()