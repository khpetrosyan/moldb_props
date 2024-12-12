import json
import tiktoken
import csv
from tqdm import tqdm

def simple_merge_csvs(input_files, output_file):
    with open(output_file, 'w', newline='') as outfile:
        is_first = True
        writer = None
        
        for filename in tqdm(input_files, desc="Merging files"):
            with open(filename) as infile:
                reader = csv.reader(infile)
                if is_first:
                    writer = csv.writer(outfile)
                    writer.writerow(next(reader))
                    is_first = False
                else:
                    next(reader)
                for row in reader:
                    writer.writerow(row)


def _normalize_column_name(col):
    return col.strip().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace(".", "_NUMBER_").replace("/", "_AND_")

def pprint_dict(d: dict):
    print(json.dumps(d, indent=2))
    
def truncate_embedding_input_to_token_limit(text, encoding, max_tokens):
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])

def calculate_tokens(s, model):
    token_encoder = tiktoken.encoding_for_model(model)
    return len(token_encoder.encode(s))

from typing import Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def get_morgan_fp_indices(smiles: str, radius: int = 2, nbits: int = 2048) -> Optional[np.ndarray]:
    """
    Get sorted indices of 1s from Morgan fingerprint of input SMILES.
    
    Args:
        smiles: Input SMILES string
        radius: Morgan fingerprint radius (default=2 for ECFP4)
        nbits: Number of bits in fingerprint (default=2048)
    
    Returns:
        Sorted array of indices where fingerprint is 1, or None if invalid SMILES
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        # Get Morgan fingerprint
        bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        
        # Get indices directly from bit info
        indices = np.array(list(bitvect.GetOnBits()), dtype=np.int32)
        indices.sort()
        
        return indices
        
    except Exception as e:
        print(f"Error generating fingerprint: {e}")
        return None


