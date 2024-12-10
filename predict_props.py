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

base_save_dir = os.path.join(ROOT_DIR, "output_props")
os.makedirs(base_save_dir, exist_ok=True)

for p in profiles:
    prop_dir = os.path.join(base_save_dir, p['name'])
    os.makedirs(prop_dir, exist_ok=True)
    p["save_dir"] = prop_dir

def get_column_names(name):
    """Get appropriate column names based on the property."""
    if name == "Cyp":
        return ["cyp_1a2", "cyp_2c19", "cyp_2c9", "cyp_2d6", "cyp_3a4"]
    else:
        return [name.lower()]
    
def predict_large(model, smiles, batch_size=64):
    """Predict properties for large sets of molecules in batches."""
    shape = model.predict(["C", "CC", "CCC"]).shape
    predictions = np.full(shape=(len(smiles), *shape[1:]), fill_value=np.nan)
    
    for i in tqdm(range(0, len(smiles), batch_size)):
        batch = smiles[i:i+batch_size]
        with redirect_stdout(StringIO()):
            batch_predictions = model.predict(batch)
        predictions[i:i+batch_size] = batch_predictions
    
    return predictions.reshape(len(smiles), -1)

def save_chunk(df, save_dir, chunk_idx, name):
    """Save a chunk of predictions to a CSV file."""
    chunk_filename = f"{name}_chunk_{chunk_idx+1}.csv"
    save_path = os.path.join(save_dir, chunk_filename)
    df.to_csv(save_path, index=False)
    print(f"Saved chunk {chunk_idx+1} of {name} predictions to {save_path}")

def get_selected_profiles(profile_arg):
    """Get the list of selected property profiles to process."""
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

def process_chunks(smiles, model, save_dir, name, chunk_size):
    """Process and save predictions in chunks."""
    num_chunks = (len(smiles) + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(smiles))
        chunk_smiles = smiles[start_idx:end_idx]
        
        print(f"\nProcessing chunk {chunk_idx+1}/{num_chunks} for {name}")
        predictions = predict_large(model, chunk_smiles)
        
        columns=get_column_names(name)
        df = pd.DataFrame(predictions, columns=columns)
        df.insert(0, "SMILES", chunk_smiles)
        save_chunk(df, save_dir, chunk_idx, name)

def main():
    parser = argparse.ArgumentParser(description='Run molecular property predictions.')
    parser.add_argument(
        '--profile',
        type=str,
        required=True,
        help='Comma-separated list of profiles to run (e.g., "LogP,LogS") or "all" to run all profiles'
    )

    parser.add_argument(
        '--chunk_size',
        type=str,
        required=False,
        default=100_000,
        help='Number of SMILES to process in each chunk'
    )
    
    args = parser.parse_args()

    chunk_size = int(args.chunk_size)
    selected_profiles = get_selected_profiles(args.profile)
    
    smiles_path = os.path.join(ROOT_DIR, "data/all_smiles.csv")
    SMILES = pd.read_csv(smiles_path).smiles.to_list()#[:1001]
    
    for p in selected_profiles:
        print(f"\nPredicting {p['name']}")
        with redirect_stdout(StringIO()):
            model = p["model"]()
        
        process_chunks(
            smiles=SMILES,
            model=model,
            save_dir=p["save_dir"],
            name=p["name"],
            chunk_size=chunk_size
        )
        
        print(f"Completed all predictions for {p['name']}")

if __name__ == "__main__":
    main()
    
"""
python predict_props.py --profile "Ames"
python predict_props.py --profile "HERG"
python predict_props.py --profile "LogD"
python predict_props.py --profile "LogP"
python predict_props.py --profile "LogS"
python predict_props.py --profile "Cyp"
"""