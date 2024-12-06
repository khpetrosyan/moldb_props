import tiktoken
import pandas as pd
from tqdm import tqdm

from utils.embeddings_utils import get_embedding
from utils.helpers import truncate_embedding_input_to_token_limit


def get_combined(df, config):
    columns = config.FEAT_COLUMNS or df.columns.to_list()
        
    encoding = tiktoken.get_encoding(config.EMBEDDING_ENCODING)
    
    def combine_and_truncate(row):
        combined_str = ""
        for col in columns:
            if config.DROPNA and pd.isna(row[col]):
                continue
            combined_str += f"\n{row[col]}"
                
        combined = " ".join(f"'{col}': '{row[col]}', " for col in columns if not pd.isna(row[col]))
        return truncate_embedding_input_to_token_limit(combined, encoding, config.N_MAX_TOKENS)
    
    return df.apply(combine_and_truncate, axis=1)

def get_emebeddings(df):
    return df.combined.progress_apply(lambda x: get_embedding(x))


def add_embeddings(config):
    df = pd.read_csv(config.csv_path)
    df["combined"] = get_combined(df, config)

    tqdm.pandas()
    df["embedding"] = get_emebeddings(df)

    df.to_csv(config.csv_with_embeddings_path)
    print(f"\n\nResutls written to {config.csv_with_embeddings_path}\n\n")
    return config.csv_with_embeddings_path
    

if __name__ == "__main__":
    from configs import SMALL_NO_NA_NO_CHIANS_CONFIG as config
    add_embeddings(config)



    