import time
import numpy as np
from rdkit import Chem
from datetime import datetime
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors

from utils.helpers import simple_merge_csvs
from tanimoto_fast import load_or_create_cache
from calculate_mol_features import main as calculate_mol_features


def get_bit_indices(bit_vect):
    """Convert RDKit bit vector to list of set bit indices."""
    return list(bit_vect.GetOnBits())

def compare_fingerprints(smiles1: str, smiles2: str):
    """Debug fingerprint differences between original and parsed formats."""
    
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        print(f"Failed to parse SMILES:\n1: {smiles1}\n2: {smiles2}")
        return None
    
    fp1_rdkit = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2_rdkit = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    
    bits1_rdkit = set(fp1_rdkit.GetOnBits())  
    bits2_rdkit = set(fp2_rdkit.GetOnBits())
    
    rdkit_sim = DataStructs.TanimotoSimilarity(fp1_rdkit, fp2_rdkit)
    
    return {
        'rdkit_bits1': bits1_rdkit,
        'rdkit_bits2': bits2_rdkit,
        'rdkit_sim': rdkit_sim,
        'rdkit_intersection': len(bits1_rdkit.intersection(bits2_rdkit)),
        'rdkit_union': len(bits1_rdkit.union(bits2_rdkit))
    }

def debug_fingerprint(smiles: str):
    """Generate and show fingerprint details for a single molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    bits = set(fp.GetOnBits())
    print(f"Generated {len(bits)} bits for SMILES: {smiles}")
    print(f"Bit positions: {sorted(list(bits))[:10]}...")
    return bits

def main(fast_search):
    print(f"Starting molecular search at {datetime.now()}")
    start_total = time.time()
    
    i_mol = np.random.randint(len(fast_search.fingerprints))
    query = fast_search.fingerprints[i_mol]
    query_smiles = fast_search.smiles[i_mol]
    
    print("\nVerifying query fingerprint...")
    rdkit_query_bits = debug_fingerprint(query_smiles)
    if rdkit_query_bits is not None:
        stored_query_bits = set(query)
        print(f"Stored bits match RDKit: {rdkit_query_bits == stored_query_bits}")
        if rdkit_query_bits != stored_query_bits:
            print("Bits in stored but not RDKit:", stored_query_bits - rdkit_query_bits)
            print("Bits in RDKit but not stored:", rdkit_query_bits - stored_query_bits)
    
    results = fast_search.search(query, k=5)
    
    print("\nQuery SMILES:", query_smiles)
    print("\nFingerprint Analysis:")
    print(f"Query fingerprint length: {len(query)} bits set")
    
    print("\nTop 5 Most Similar Molecules:")
    print("Rank\tFast-Sim\tRDKit-Sim\tIntersection\tUnion\tBits_Match\tSMILES")
    
    for rank, (idx, fast_sim) in enumerate(results, 1):
        target_smiles = fast_search.smiles[idx]
        target = fast_search.fingerprints[idx]
        
        
        debug_info = compare_fingerprints(query_smiles, target_smiles)
        if debug_info is None:
            continue
        
        fast_bits1 = set(query)
        fast_bits2 = set(target)
        
        bits_match = (fast_bits1 == debug_info['rdkit_bits1'] and 
                     fast_bits2 == debug_info['rdkit_bits2'])
        
        print(f"{rank}\t{fast_sim:.3f}\t{debug_info['rdkit_sim']:.3f}\t"
              f"{len(fast_bits1.intersection(fast_bits2))}/{debug_info['rdkit_intersection']}\t"
              f"{len(fast_bits1.union(fast_bits2))}/{debug_info['rdkit_union']}\t"
              f"{bits_match}\t{target_smiles}")
        
        if rank == 1:  
            print("\nDetailed analysis of first result:")
            print("Fast Search bits in query but not RDKit:", fast_bits1 - debug_info['rdkit_bits1'])
            print("RDKit bits in query but not Fast Search:", debug_info['rdkit_bits1'] - fast_bits1)
            print("Fast Search bits in target but not RDKit:", fast_bits2 - debug_info['rdkit_bits2'])
            print("RDKit bits in target but not Fast Search:", debug_info['rdkit_bits2'] - fast_bits2)
    
    total_time = time.time() - start_total
    print(f"\nTotal runtime: {total_time:.2f} seconds")
    print(f"Finished at {datetime.now()}")

if __name__ == "__main__":
    csv_file = '/Users/khorenpetrosyan/moldb_props/output_mol_feature/molecular_features.csv'
    fast_search, load_time = load_or_create_cache(csv_file)
    for _ in range(20):
        print("\n", "="*80, "\n")
        main(fast_search=fast_search)