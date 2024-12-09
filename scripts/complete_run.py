import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from embed import add_embeddings
from preprocess_data import process_bindingdb_tsv
from configs import TSV_PATH, NORMALIZED_FEAT_COLUMNS, ROOT_DIR


NORMALIZED_FEAT_COLUMNS = [
    'Article_DOI',
    'Authors',
    'ZINC_ID_of_Ligand',
    'koff_s-1',
    'kon_M-1-s-1',
    'pH',
    'BindingDB_Entry_DOI',
    'BindingDB_Ligand_Name',
    'BindingDB_MonomerID',
    'BindingDB_Reactant_set_id',
    'ChEBI_ID_of_Ligand',
    'ChEMBL_ID_of_Ligand',
    'Curation_AND_DataSource',
    'DrugBank_ID_of_Ligand',
    'EC50_nM',
    'IC50_nM',
    'IUPHAR_GRAC_ID_of_Ligand',
    'Institution',
    'KEGG_ID_of_Ligand',
    'Kd_nM',
    'Ki_nM',
    'Ligand_HET_ID_in_PDB',
    'Ligand_InChI',
    'Ligand_InChI_Key',
    'Ligand_SMILES',
    'Link_to_Ligand_in_BindingDB',
    'Link_to_Ligand-Target_Pair_in_BindingDB',
    'Link_to_Target_in_BindingDB',
    'Number_of_Protein_Chains_in_Target_>1_implies_a_multichain_complex',
    'PDB_IDs_for_Ligand-Target_Complex',
    'PMID',
    'Patent_Number',
    'PubChem_AID',
    'PubChem_CID',
    'PubChem_SID',
    'Target_Name',
    'Target_Source_Organism_According_to_Curator_or_DataSource',
    'Temp_C',
    'PDB_IDs_of_Target_Chain',
    'UniProt_SwissProt_Entry_Name_of_Target_Chain',
    'UniProt_SwissProt_Primary_ID_of_Target_Chain',
    'UniProt_TrEMBL_Submitted_Name_of_Target_Chain',
    'UniProt_SwissProt_Recommended_Name_of_Target_Chain',
    'UniProt_TrEMBL_Entry_Name_of_Target_Chain',
    'UniProt_TrEMBL_Primary_ID_of_Target_Chain',
    
]

EXTRA_COLUMNS = [
    'BindingDB_Target_Chain_Sequence',
    'BindingDB_Target_Chain_Sequence_NUMBER_1',
    'BindingDB_Target_Chain_Sequence_NUMBER_2',
    'BindingDB_Target_Chain_Sequence_NUMBER_3',
    'PDB_IDs_of_Target_Chain_NUMBER_1',
    'PDB_IDs_of_Target_Chain_NUMBER_2',
    'PDB_IDs_of_Target_Chain_NUMBER_3',
    'UniProt_SwissProt_Alternative_IDs_of_Target_Chain',
    'UniProt_SwissProt_Alternative_IDs_of_Target_Chain_NUMBER_1',
    'UniProt_SwissProt_Alternative_IDs_of_Target_Chain_NUMBER_2',
    'UniProt_SwissProt_Alternative_IDs_of_Target_Chain_NUMBER_3',
    'UniProt_SwissProt_Entry_Name_of_Target_Chain_NUMBER_1',
    'UniProt_SwissProt_Entry_Name_of_Target_Chain_NUMBER_2',
    'UniProt_SwissProt_Entry_Name_of_Target_Chain_NUMBER_3',
    'UniProt_SwissProt_Primary_ID_of_Target_Chain_NUMBER_1',
    'UniProt_SwissProt_Primary_ID_of_Target_Chain_NUMBER_2',
    'UniProt_SwissProt_Primary_ID_of_Target_Chain_NUMBER_3',
    'UniProt_SwissProt_Recommended_Name_of_Target_Chain_NUMBER_1',
    'UniProt_SwissProt_Recommended_Name_of_Target_Chain_NUMBER_2',
    'UniProt_SwissProt_Recommended_Name_of_Target_Chain_NUMBER_3',
    'UniProt_SwissProt_Secondary_IDs_of_Target_Chain',
    'UniProt_SwissProt_Secondary_IDs_of_Target_Chain_NUMBER_1',
    'UniProt_SwissProt_Secondary_IDs_of_Target_Chain_NUMBER_2',
    'UniProt_SwissProt_Secondary_IDs_of_Target_Chain_NUMBER_3',
    'UniProt_TrEMBL_Alternative_IDs_of_Target_Chain',
    'UniProt_TrEMBL_Alternative_IDs_of_Target_Chain_NUMBER_1',
    'UniProt_TrEMBL_Alternative_IDs_of_Target_Chain_NUMBER_2',
    'UniProt_TrEMBL_Alternative_IDs_of_Target_Chain_NUMBER_3',
    'UniProt_TrEMBL_Entry_Name_of_Target_Chain_NUMBER_1',
    'UniProt_TrEMBL_Entry_Name_of_Target_Chain_NUMBER_2',
    'UniProt_TrEMBL_Entry_Name_of_Target_Chain_NUMBER_3',
    'UniProt_TrEMBL_Primary_ID_of_Target_Chain_NUMBER_1',
    'UniProt_TrEMBL_Primary_ID_of_Target_Chain_NUMBER_2',
    'UniProt_TrEMBL_Primary_ID_of_Target_Chain_NUMBER_3',
    'UniProt_TrEMBL_Secondary_IDs_of_Target_Chain',
    'UniProt_TrEMBL_Secondary_IDs_of_Target_Chain_NUMBER_1',
    'UniProt_TrEMBL_Secondary_IDs_of_Target_Chain_NUMBER_2',
    'UniProt_TrEMBL_Secondary_IDs_of_Target_Chain_NUMBER_3',
    'UniProt_TrEMBL_Submitted_Name_of_Target_Chain_NUMBER_1',
    'UniProt_TrEMBL_Submitted_Name_of_Target_Chain_NUMBER_2',
    'UniProt_TrEMBL_Submitted_Name_of_Target_Chain_NUMBER_3',
]

class ExperimentTSNEExtraCols:
    name = __qualname__
    DROPNA = True
    n_rows = 500
    up_to_n_chains = 0
    GPT_MODEL = "gpt-4"
    TSV_PATH = TSV_PATH
    N_MAX_TOKENS = 16_000
    TOKEN_BUDGET = 32_000
    EMBEDDING_ENCODING = "cl100k_base"
    FEAT_COLUMNS = NORMALIZED_FEAT_COLUMNS+EXTRA_COLUMNS
    EMBEDDING_MODEL = "text-embedding-3-small"
    OUTPUTS_DIR = os.path.join(ROOT_DIR, f'output_data_{__qualname__}')
    
    csv_path = os.path.join(OUTPUTS_DIR, f'bdb_{n_rows}_up_to_{up_to_n_chains}_chains.csv')
    csv_with_embeddings_path = csv_path.replace('_chains.csv', '_chains_with_embeddings.csv')


def main(config):
    process_bindingdb_tsv(config)
    add_embeddings(config)

    df = pd.read_csv(config.csv_with_embeddings_path)

    matrix = np.array(df.embedding.apply(ast.literal_eval).to_list())

    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
    vis_dims = tsne.fit_transform(matrix)
    print(vis_dims.shape)



    x = vis_dims[:, 0]
    y = vis_dims[:, 1]

    plt.scatter(x, y, alpha=0.3) 
    plt.savefig(os.path.join(config.OUTPUTS_DIR, f"tsne_{config.name}.png"))
    plt.show()
    
if __name__ == '__main__':
    main(config=ExperimentTSNEExtraCols)