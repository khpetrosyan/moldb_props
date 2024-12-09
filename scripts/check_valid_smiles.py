from contextlib import redirect_stdout
from io import StringIO
import logging
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm


logging.getLogger().setLevel(logging.ERROR)
for logger_name in ["rdkit", "molprops", "*"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)


smiles = pd.read_csv("/home/khoren/moldb_props/data/all_smiles.csv").smiles.to_list()
smiles = np.random.choice(smiles, 100_000)

inv = 0
for s in tqdm(smiles):
    with redirect_stdout(StringIO()):
        if not Chem.MolFromSmiles(s):
            inv += 1
            
print(f"Invalid smiles: {inv}/{len(smiles)} or {inv/len(smiles)*100:.2f}%")