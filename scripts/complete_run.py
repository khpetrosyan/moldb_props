import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from embed import add_embeddings
from preprocess_data import process_bindingdb_tsv
from configs import TSV_PATH, NORMALIZED_FEAT_COLUMNS, ROOT_DIR


class ExperimentTSNE:
    name = __qualname__
    DROPNA = True
    n_rows = 500
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
    main(config=ExperimentTSNE)