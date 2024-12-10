import os
import sys
import logging
import argparse
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
][::-1]

save_dir = os.path.join(ROOT_DIR, "output_props")
os.makedirs(save_dir, exist_ok=True)

for p in profiles:
    p["save_path"] = os.path.join(save_dir, f"{p['name']}.csv")

def predict_large(model, smiles, batch_size=64):
    shape = model.predict(["C", "CC", "CCC"]).shape
    predictions = np.full(shape=(len(smiles), *shape[1:]), fill_value=np.nan)
    
    for i in tqdm(range(0, len(smiles), batch_size)):
        batch = smiles[i:i+batch_size]
        with redirect_stdout(StringIO()):
            batch_predictions = model.predict(batch)
        predictions[i:i+batch_size] = batch_predictions
    return predictions.reshape(len(smiles), -1)

def get_selected_profiles(profile_arg):
    if profile_arg.lower() == 'all':
        return profiles
    
    requested_profiles = [p.strip() for p in profile_arg.split(',')]
    available_names = [p['name'] for p in profiles]
    
    invalid_profiles = [p for p in requested_profiles if p not in available_names]
    if invalid_profiles:
        print(f"Error: Invalid profile names: {invalid_profiles}")
        print(f"Available profiles: {available_names}")
        sys.exit(1)
        
    return [p for p in profiles if p['name'] in requested_profiles]

def main():
    parser = argparse.ArgumentParser(description='Run molecular property predictions.')
    parser.add_argument('--profile', 
                       type=str,
                       required=True,
                       help='Comma-separated list of profiles to run (e.g., "LogP,LogS") or "all" to run all profiles')
    
    args = parser.parse_args()
    selected_profiles = get_selected_profiles(args.profile)
    
    smiles_path = os.path.join(ROOT_DIR, "data/all_smiles.csv")
    SMILES = pd.read_csv(smiles_path).smiles.to_list()#[:1280]
    
    for p in selected_profiles:
        print(f"Predicting {p['name']}")
        with redirect_stdout(StringIO()):
            model = p["model"]()
        
        name = p["name"]
        save_path = p["save_path"]
        
        predictions = predict_large(model, SMILES)
        df = pd.DataFrame(predictions, columns=[f"{name}_{i+1}" for i in range(predictions.shape[1])])
        df.insert(0, "SMILES", SMILES)
        df.to_csv(save_path, index=False)
        print(f"Saved {name} predictions to {save_path}")

if __name__ == "__main__":
    main()