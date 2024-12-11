import os
import logging
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, DataStructs
from contextlib import redirect_stdout
from io import StringIO
from tqdm import tqdm

from configs import ROOT_DIR
from utils.helpers import simple_merge_csvs

logging.getLogger().setLevel(logging.ERROR)
for logger_name in ["rdkit", "*"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


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
    try:
        with redirect_stdout(StringIO()):
            molecule = Chem.MolFromSmiles(smiles)
            
            if molecule is None:
                return
            
            results = {
                'SMILES': smiles,
                'Morgan_Fingerprint': '',
                'Fingerprint_Sum': np.nan,
                'Topological_Polar_Surface_Area': np.nan,
                'Molecular_Weight': np.nan,
                'Molar_Refractivity': np.nan,
                'Complexity_Index': np.nan,
                'LogP': np.nan,
                'Balaban_J_Index': np.nan,
                'Ring_Count': np.nan,
                'Surface_Area': np.nan,
                'H_Bond_Donors': np.nan,
                'H_Bond_Acceptors': np.nan,
                'Heteroatom_Count': np.nan,
                'Heavy_Atom_Count': np.nan,
                'Aromatic_Ring_Count': np.nan,
                'Rotatable_Bond_Count': np.nan,
                'Saturated_Ring_Count': np.nan,
                'Aliphatic_Ring_Count': np.nan,
                'Valence_Electron_Count': np.nan,
                'SP2_Atom_Count': np.nan,
                'Basic_Nitrogen_Count': np.nan,
                'Has_Sulfonylurea': np.nan,
                'Has_Halogenated_Phenyl': np.nan
            }
            
            for fg_smarts, fg_name in FRAGMENT_NAMES.items():
                results[f'Has_{fg_name}'] = np.nan
                
            fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
            fingerprint_indices = sorted(list(fingerprint.GetOnBits()))
            results['Fingerprint_Sum'] = len(fingerprint_indices)
            results['Morgan_Fingerprint'] = fingerprint_indices
            
            results.update({
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
                'SP2_Atom_Count': sum(1 for atom in molecule.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP2),
                'Basic_Nitrogen_Count': sum(1 for atom in molecule.GetAtoms() if atom.GetSymbol() == 'N' and atom.GetTotalDegree() == 3),
                'Has_Sulfonylurea': molecule.HasSubstructMatch(Chem.MolFromSmarts("[#16](=[O])(=[O])-[#7]")),
                'Has_Halogenated_Phenyl': molecule.HasSubstructMatch(Chem.MolFromSmarts("c1ccccc1[F,Cl,Br,I]"))
            })
            
            fragment_results = fragments(molecule)
            for fg_smarts, fg_name in FRAGMENT_NAMES.items():
                idx = fgs.index(fg_smarts)
                results[f'Has_{fg_name}'] = fragment_results[idx]
                
            return results
    
    except Exception:
        logger.exception(f"Error processing SMILES {smiles}: \n\n")
        return

def process_smiles_chunk(smiles_list, chunk_index, output_dir):
    results = []
    for smiles in tqdm(smiles_list, desc=f"Processing chunk {chunk_index}"):
        if result := calculate_molecular_features(smiles):
            results.append(result)
    
    if results:
        df = pd.DataFrame(results)
        df['Morgan_Fingerprint'] = df['Morgan_Fingerprint'].apply(lambda x: ','.join(map(str, x)))

        output_file = os.path.join(output_dir, f'molecular_features_chunk_{chunk_index:04d}.csv')
        df.to_csv(output_file, index=False)
        print(f"Saved chunk {chunk_index} to {output_file}")
        return len(results), output_file
    return 0, output_file

def main(N=None, chunk_size=100_000, save_by_chunks=False):
    input_file = os.path.join(ROOT_DIR, 'data/all_smiles.csv')
    output_dir = os.path.join(ROOT_DIR, 'output_mol_feature')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading SMILES from: {input_file}")
    
    df = pd.read_csv(input_file)
    
    smiles_list = df['smiles'].tolist()
    if not isinstance(N, int) or not N > 0:
        N = len(smiles_list)
        
    smiles_list = smiles_list[:N]
    
    total_processed = 0
    chunk_size = min(100_000, N)
    
    output_files = []
    for i in range(0, len(smiles_list), chunk_size):
        chunk = smiles_list[i:i+chunk_size]
        chunk_index = i // chunk_size
        processed, output_file = process_smiles_chunk(chunk, chunk_index, output_dir)
        total_processed += processed
        output_files.append(output_file)
    
    if not save_by_chunks:
        simple_merge_csvs(output_files, os.path.join(output_dir, 'molecular_features.csv'))
        [os.remove(f) for f in output_files]
    
    print(f"\nProcessing complete!")
    print(f"Total SMILES processed: {total_processed}")
    print(f"Output files saved in: {output_dir}")
    
    return output_files

if __name__ == "__main__":
    main()