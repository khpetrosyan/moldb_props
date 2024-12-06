import os
import csv
import logging
import pandas as pd

from columns import NORMALIZED_COLUMNS, ORIGINAL_COLUMNS


def process_bindingdb_tsv(config) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Loading BindingDB TSV file...")
    smaller_TSV_PATH = config.TSV_PATH.replace('.tsv', f'_{config.n_rows}.tsv')
    
    os.system(f"head -n 1 {config.TSV_PATH} > {smaller_TSV_PATH}")
    os.system(f"cat {config.TSV_PATH} | shuf | head -n {config.n_rows} >> {smaller_TSV_PATH}")
    
    try:
        df = pd.read_csv(smaller_TSV_PATH, 
                        sep='\t',
                        low_memory=False,
                        on_bad_lines='skip',
                        quoting=csv.QUOTE_NONE,  # No quote handling
                        dtype=str,  # Read everything as string
                        encoding='utf-8',
                        encoding_errors='ignore',  # Handle any encoding issues
                        )
        
        logger.info(f"Successfully loaded {len(df)} rows")
        
        df = df[ORIGINAL_COLUMNS]
        df.columns = NORMALIZED_COLUMNS

        data_dir = os.path.dirname(config.csv_path)
        os.makedirs(data_dir, exist_ok=True)
        
        df.to_csv(
            config.csv_path, 
            index=False,
            quoting=csv.QUOTE_MINIMAL,  # Only quote fields that contain commas or quotes
            encoding='utf-8'
        )
        
        os.system(f"rm {smaller_TSV_PATH}")
        
        
    except Exception:
        logger.exception(f"Failed to read file normally\n")



if __name__ == '__main__':
    from configs import SMALL_NO_NA_NO_CHIANS_CONFIG as config
    process_bindingdb_tsv(config)
    



