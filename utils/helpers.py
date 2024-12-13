import csv
import json
import tiktoken
from tqdm import tqdm
from numba import jit


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



@jit(nopython=True)
def count_common_elements(arr1, arr2):
    last_found = -1  
    count = 0
    i = 0 
    j = 0
    
    while i < len(arr1) and j < len(arr2):
        if arr1[i] < arr2[j]:
            i += 1
        elif arr1[i] > arr2[j]:
            j += 1
        else: 
            if arr1[i] != last_found:
                count += 1
                last_found = arr1[i]
            i += 1
            j += 1
    
    return count

@jit(nopython=True)
def tanimoto_similarity(arr1, arr2):
    intersection = count_common_elements(arr1, arr2)
    unique1 = len(arr1)
    unique2 = len(arr2)
    union = unique1 + unique2 - intersection
    
    if union == 0:
        return 1.0 if len(arr1) == 0 and len(arr2) == 0 else 0.0
        
    return intersection / union

assert tanimoto_similarity(np.array([1, 2, 3]), np.array([2, 3, 4])) == 0.5; "Tanimoto calculator broken!" # to compile the function and test it