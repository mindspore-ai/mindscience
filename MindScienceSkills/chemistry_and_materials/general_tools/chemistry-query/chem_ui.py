import gradio as gr
import subprocess
import json
import os
import pandas as pd
from PIL import Image

WORK_DIR = "/home/democritus/.openclaw/workspace/skills/chemistry-query/scripts"

def analyze(smiles):
    # Props
    proc = subprocess.run(["python3", "rdkit_mol.py", "--smiles", smiles, "--action", "props"], cwd=WORK_DIR, capture_output=True, text=True)
    props = json.loads(proc.stdout)
    
    # Draw PNG
    subprocess.run(["python3", "rdkit_mol.py", "--smiles", smiles, "--action", "draw", "--output", "mol.png"], cwd=WORK_DIR)
    img = Image.open(os.path.join(WORK_DIR, "mol.png"))
    
    # ADMET
    proc = subprocess.run(["python3", "admet_predict.py", "--smiles", smiles], cwd=WORK_DIR, capture_output=True, text=True)
    admet = json.loads(proc.stdout)
    
    return props, img, admet

iface = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(label="SMILES", value="CCO"),
    outputs=[gr.JSON(label="Props"), gr.Image(label="2D Viz"), gr.JSON(label="ADMET")],
    title="Chemistry Query Agent 🧪",
    description="PubChem/RDKit analysis"
)

if __name__ == "__main__":
    iface.launch(share=True)