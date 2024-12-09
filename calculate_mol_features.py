import os
import logging
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, DataStructs
from contextlib import redirect_stdout
from io import StringIO
from tqdm import tqdm


logging.getLogger().setLevel(logging.ERROR)
for logger_name in ["rdkit", "*"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)


FRAGMENT_NAMES = {
    "o1cccc1": "furan_ring",
    "[CX3H1](=O)[#6]": "aldehyde",
    "[#6][CX3](=O)[OX2H0][#6]": "ester",
    "[#6][CX3](=O)[#6]": "ketone",
    "C(=O)-N": "amide",
    "[SH]": "thiol",
    "[OH]": "hydroxyl",
    "*-[N;D2]-[C;D3](=O)-[C;D1;H3]": "n_acetyl",
    "*-C(=O)[O;D1]": "carboxylic_acid",
    "*-C(=O)[O;D2]-[C;D1;H3]": "methyl_ester",
    "*-C(=O)-[C;D1]": "acyl",
    "*-C(=O)-[N;D1]": "primary_amide",
    "*-C(=O)-[C;D1;H3]": "acetyl",
    "*-[N;D2]=[C;D2]=[O;D1]": "isocyanate",
    "*-[N;D2]=[C;D2]=[S;D1]": "isothiocyanate",
    "*-[N;D3](=[O;D1])[O;D1]": "nitro",
    "*-[N;R0]=[O;D1]": "nitroso",
    "*=[N;R0]-[O;D1]": "oxime",
    "*-[N;R0]=[C;D1;H2]": "imine",
    "*-[N;D2]=[N;D2]-[C;D1;H3]": "hydrazone",
    "*-[N;D2]=[N;D1]": "diazo",
    "*-[N;D2]#[N;D1]": "azide",
    "*-[C;D2]#[N;D1]": "nitrile",
    "*-[S;D4](=[O;D1])(=[O;D1])-[N;D1]": "sulfonamide_primary",
    "*-[N;D2]-[S;D4](=[O;D1])(=[O;D1])-[C;D1;H3]": "n_methyl_sulfonamide",
    "*-[S;D4](=O)(=O)-[O;D1]": "sulfonic_acid",
    "*-[S;D4](=O)(=O)-[O;D2]-[C;D1;H3]": "methyl_sulfonate",
    "*-[S;D4](=O)(=O)-[C;D1;H3]": "methyl_sulfone",
    "*-[S;D4](=O)(=O)-[Cl]": "sulfonyl_chloride",
    "*-[S;D3](=O)-[C;D1]": "sulfoxide",
    "*-[S;D2]-[C;D1;H3]": "methyl_sulfide",
    "*-[S;D1]": "thiol",
    "*=[S;D1]": "thione",
    "*-[#9,#17,#35,#53]": "halogen",
    "*-[C;D4]([C;D1])([C;D1])-[C;D1]": "t_butyl",
    "*-[C;D4](F)(F)F": "trifluoromethyl",
    "*-[C;D2]#[C;D1;H]": "terminal_alkyne",
    "*-[C;D3]1-[C;D2]-[C;D2]1": "cyclopropyl",
    "*-[O;D2]-[C;D2]-[C;D1;H3]": "ethoxy",
    "*-[O;D2]-[C;D1;H3]": "methoxy",
    "*-[O;D1]": "hydroxy",
    "*=[O;D1]": "carbonyl",
    "*-[N;D1]": "primary_amine",
    "*#[N;D1]": "nitrile"
}


fgs = list(FRAGMENT_NAMES.keys())

def fragments(molecule):
    """Calculate molecular fragments"""
    def _is_fg_in_mol(fg):
        fgmol = Chem.MolFromSmarts(fg)
        return len(Chem.Mol.GetSubstructMatches(molecule, fgmol, uniquify=True)) > 0
    return [_is_fg_in_mol(fg) for fg in fgs]

def calculate_molecular_features(smiles):
    """
    Calculate molecular features for a given SMILES string.
    Returns a dictionary containing the fingerprint vector, its sum, and other molecular descriptors.
    """
    try:
        with redirect_stdout(StringIO()):
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                return None
            
            
            fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
            fingerprint_array = np.zeros((2048,))
            DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
            
            
            results = {
                'SMILES': smiles,
                'Morgan_Fingerprint': fingerprint_array,
                'Fingerprint_Sum': int(np.sum(fingerprint_array)),
                'Topological_Polar_Surface_Area': Descriptors.TPSA(molecule),
                'Molecular_Weight': Descriptors.MolWt(molecule),
                'Molar_Refractivity': Descriptors.MolMR(molecule),
                'Complexity_Index': Descriptors.BertzCT(molecule),
                'LogP': Descriptors.MolLogP(molecule),
                'Balaban_J_Index': Descriptors.BalabanJ(molecule),
                'Ring_Count': Descriptors.RingCount(molecule),
                'Surface_Area': Descriptors.LabuteASA(molecule),
                'H_Bond_Donors': Descriptors.NumHDonors(molecule),
                'H_Bond_Acceptors': Descriptors.NumHAcceptors(molecule),
                'Heteroatom_Count': Descriptors.NumHeteroatoms(molecule),
                'Heavy_Atom_Count': Descriptors.HeavyAtomCount(molecule),
                'Aromatic_Ring_Count': Descriptors.NumAromaticRings(molecule),
                'Rotatable_Bond_Count': Descriptors.NumRotatableBonds(molecule),
                'Saturated_Ring_Count': Descriptors.NumSaturatedRings(molecule),
                'Aliphatic_Ring_Count': Descriptors.NumAliphaticRings(molecule),
                'Valence_Electron_Count': Descriptors.NumValenceElectrons(molecule),
                'SP2_Atom_Count': sum(1 for atom in molecule.GetAtoms() 
                                    if atom.GetHybridization() == Chem.HybridizationType.SP2),
                'Basic_Nitrogen_Count': sum(1 for atom in molecule.GetAtoms() 
                                          if atom.GetSymbol() == 'N' and atom.GetTotalDegree() == 3),
                'Has_Sulfonylurea': molecule.HasSubstructMatch(Chem.MolFromSmarts("[#16](=[O])(=[O])-[#7]")),
                'Has_Halogenated_Phenyl': molecule.HasSubstructMatch(Chem.MolFromSmarts("c1ccccc1[F,Cl,Br,I]"))
            }
            
            
            fragment_results = fragments(molecule)
            for fg_smarts, fg_name in FRAGMENT_NAMES.items():
                idx = fgs.index(fg_smarts)
                results[f'Has_{fg_name}'] = fragment_results[idx]
                
            return results
    
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {str(e)}")
        return None

def process_smiles_chunk(smiles_list, chunk_index, output_dir):
    """Process a chunk of SMILES and save to a separate file"""
    results = []
    for smiles in tqdm(smiles_list, desc=f"Processing chunk {chunk_index}"):
        result = calculate_molecular_features(smiles)
        if result is not None:
            results.append(result)
    
    if results:
        df = pd.DataFrame(results)
        
        df['Morgan_Fingerprint'] = df['Morgan_Fingerprint'].apply(lambda x: ','.join(map(str, x)))
        
        
        output_file = os.path.join(output_dir, f'molecular_features_chunk_{chunk_index:04d}.csv')
        df.to_csv(output_file, index=False)
        print(f"Saved chunk {chunk_index} to {output_file}")
        return len(results)
    return 0

def main():
    
    input_file = '/home/khoren/moldb_props/data/all_smiles.csv'
    output_dir = '/home/khoren/moldb_props/output_mol_feature'
    
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading SMILES from: {input_file}")
    
    
    df = pd.read_csv(input_file)
    smiles_list = df['smiles'].tolist()#[:100]  
    
    
    chunk_size = 100000
    total_processed = 0
    
    for i in range(0, len(smiles_list), chunk_size):
        chunk = smiles_list[i:i+chunk_size]
        chunk_index = i // chunk_size
        processed = process_smiles_chunk(chunk, chunk_index, output_dir)
        total_processed += processed
    
    print(f"\nProcessing complete!")
    print(f"Total SMILES processed: {total_processed}")
    print(f"Output files saved in: {output_dir}")

if __name__ == "__main__":
    main()