import os
from columns import NORMALIZED_FEAT_COLUMNS

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

TSV_PATH = os.path.join(ROOT_DIR, 'data/' 'BindingDB_All.tsv')


class DEFAULT_CONFIG:
    DROPNA = False
    n_rows = 20_000
    up_to_n_chains = 3
    GPT_MODEL = "gpt-4"
    TSV_PATH = TSV_PATH
    N_MAX_TOKENS = 16_000
    TOKEN_BUDGET = 4096 - 500
    EMBEDDING_ENCODING = "cl100k_base"
    FEAT_COLUMNS = NORMALIZED_FEAT_COLUMNS
    EMBEDDING_MODEL = "text-embedding-3-small"
    OUTPUTS_DIR = os.path.join(ROOT_DIR, f'output_data_{__qualname__}')
    
    csv_path = os.path.join(OUTPUTS_DIR, f'bdb_{n_rows}_up_to_{up_to_n_chains}_chains.csv')
    csv_with_embeddings_path = csv_path.replace('_chains.csv', '_chains_with_embeddings.csv')



class SMALL_WITH_NA_CONFIG:
    name = __qualname__
    DROPNA = False
    n_rows = 100
    up_to_n_chains = 3
    GPT_MODEL = "gpt-4"
    TSV_PATH = TSV_PATH
    N_MAX_TOKENS = 16_000
    TOKEN_BUDGET = 4096 - 500
    EMBEDDING_ENCODING = "cl100k_base"
    FEAT_COLUMNS = NORMALIZED_FEAT_COLUMNS
    EMBEDDING_MODEL = "text-embedding-3-small"
    OUTPUTS_DIR = os.path.join(ROOT_DIR, f'output_data_{__qualname__}')
    
    csv_path = os.path.join(OUTPUTS_DIR, f'bdb_{n_rows}_up_to_{up_to_n_chains}_chains.csv')
    csv_with_embeddings_path = csv_path.replace('_chains.csv', '_chains_with_embeddings.csv')


class SMALL_NO_NA_CONFIG:
    DROPNA = True
    n_rows = 100
    up_to_n_chains = 3
    GPT_MODEL = "gpt-4"
    TSV_PATH = TSV_PATH
    N_MAX_TOKENS = 16_000
    TOKEN_BUDGET = 4096 - 500
    EMBEDDING_ENCODING = "cl100k_base"
    FEAT_COLUMNS = NORMALIZED_FEAT_COLUMNS
    EMBEDDING_MODEL = "text-embedding-3-small"
    OUTPUTS_DIR = os.path.join(ROOT_DIR, f'output_data_{__qualname__}')
    
    csv_path = os.path.join(OUTPUTS_DIR, f'bdb_{n_rows}_up_to_{up_to_n_chains}_chains.csv')
    csv_with_embeddings_path = csv_path.replace('_chains.csv', '_chains_with_embeddings.csv')


class SMALL_NO_NA_NO_CHIANS_CONFIG:
    name = __qualname__
    DROPNA = True
    n_rows = 100
    up_to_n_chains = 0
    GPT_MODEL = "gpt-4"
    TSV_PATH = TSV_PATH
    N_MAX_TOKENS = 16_000
    TOKEN_BUDGET = 32_000
    EMBEDDING_ENCODING = "cl100k_base"
    FEAT_COLUMNS = NORMALIZED_FEAT_COLUMNS
    EMBEDDING_MODEL = "text-embedding-3-small"
    OUTPUTS_DIR = os.path.join(ROOT_DIR, f'output_data_{__qualname__}')
    
    csv_path = os.path.join(OUTPUTS_DIR, f'bdb_{n_rows}_up_to_{up_to_n_chains}_chains.csv')
    csv_with_embeddings_path = csv_path.replace('_chains.csv', '_chains_with_embeddings.csv')
    
