import os
import sys
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from molprops import LogS, LogP, LogD, HERG, Cyp, Ames
from contextlib import redirect_stdout
from io import StringIO

from configs import ROOT_DIR


logging.getLogger().setLevel(logging.ERROR)
for logger_name in ["rdkit", "molprops", "*"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)


profiles = [
    {"name": "Cyp", "model": Cyp},
    {"name": "LogS", "model": LogS},
    {"name": "LogP", "model": LogP},
    {"name": "LogD", "model": LogD},
    {"name": "HERG", "model": HERG},
    {"name": "Ames", "model": Ames},
]

save_dir = os.path.join(ROOT_DIR, "output_props")
os.makedirs(save_dir, exist_ok=True)

for p in profiles:
    p["save_path"] = os.path.join(save_dir, f"{p['name']}.csv")

def predict_large(name, model, smiles, batch_size=640):
    shape = (len(smiles), 5 if name == "Cyp" else 1)
    predictions = np.full(shape=shape, fill_value=1)#np.nan)
    
    for i in tqdm(range(0, len(smiles), batch_size)):
        batch = smiles[i:i+batch_size]
        with redirect_stdout(StringIO()):
            batch_predictions = model.predict(batch)
        predictions[i:i+batch_size] = batch_predictions
    return predictions.reshape(len(smiles), -1) 

def main():
    smiles_path = os.path.join(ROOT_DIR, "data/all_smiles.csv")
    SMILES = pd.read_csv(smiles_path).smiles.to_list()
    
    for p in profiles:
        print(f"Predicting {p['name']}")
        with redirect_stdout(StringIO()):
            model = p["model"]()
        
        name = p["name"]
        save_path = p["save_path"]  
        
        predictions = predict_large(name, model, SMILES)
        df = pd.DataFrame(predictions, columns=[f"{name}_{i+1}" for i in range(predictions.shape[1])])
        df.insert(0, "SMILES", SMILES)
        df.to_csv(save_path, index=False)
        print(f"Saved {name} predictions to {save_path}")
        
if __name__ == "__main__":
    main()