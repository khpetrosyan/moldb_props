import os
import heapq
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor

from utils.helpers import get_morgan_fp_indices, tanimoto_similarity

import logging
logger = logging.getLogger(__name__)


def parse_fingerprint(fp_str: str) -> Optional[np.ndarray]:
    if pd.isna(fp_str) or not isinstance(fp_str, str):
        return None
    try:
        return np.fromstring(fp_str.strip('"'), sep=',', dtype=np.int32)
    except (ValueError, AttributeError):
        return None


def batch_parse_fingerprints(chunk):
    return [parse_fingerprint(fp) for fp in chunk if not pd.isna(fp)]


class FastSearch:
    def __init__(self, csv_path=None, fingerprint_cache_dir=None, verbose=True):
        self.smiles = None
        self.verbose = verbose
        self.fp_lengths = None
        self.fingerprints = None
        self.fingerprint_cache_dir = Path(fingerprint_cache_dir or "fingerprint_cache")
        
        if csv_path:
            self._initialize_from_source(csv_path)
    
    def _log_info(self, s):
        if self.verbose:
            logger.info(s)
    
    def _initialize_from_source(self, csv_path: str):
        """Initialize the searcher either from cache or by creating new cache."""
        CACHE_DIR = self.fingerprint_cache_dir
        FP_CACHE = CACHE_DIR / "fingerprints.npy"
        METADATA_CACHE = CACHE_DIR / "metadata.pkl"

        CACHE_DIR.mkdir(exist_ok=True)
        csv_mtime = os.path.getmtime(csv_path)
        cache_exists = all(p.exists() for p in [FP_CACHE, METADATA_CACHE])
        
        if cache_exists:
            with open(METADATA_CACHE, 'rb') as f:
                try:
                    import pickle
                    metadata = pickle.load(f)
                    if metadata.get('csv_mtime') == csv_mtime:
                        self._load_from_cache()
                        df = pd.read_csv(csv_path)
                        self.fp_lengths = df['Fingerprint_Sum'].values
                        logger.info(f"Loaded {len(self.fingerprints):,} fingerprints from cache")
                        return
                except Exception as e:
                    logger.warning(f"Failed to load cache: {e}")
        
        logger.info("Cache not found or outdated. Creating new cache...")        
        self._create_new_cache(csv_path)
    
    def _create_new_cache(self, csv_path: str):
        """Create new cache from CSV file."""
        df = pd.read_csv(csv_path)        
        n_cores = multiprocessing.cpu_count()
        chunk_size = max(1, len(df) // (n_cores - 2))  # Leave 2 cores for other tasks
        chunks = [df['Morgan_Fingerprint'][i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        
        fingerprints = []
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            futures = [executor.submit(batch_parse_fingerprints, chunk) for chunk in chunks]
            for future in tqdm(futures, desc="Processing chunks"):
                batch_fps = future.result()
                fingerprints.extend(batch_fps)
        
        smiles = df['SMILES'].tolist()
        fp_lengths = df['Fingerprint_Sum'].tolist()
        
        self.fit(fingerprints, fp_lengths, smiles)
        self._save_cache()
        
        with open(self.fingerprint_cache_dir / "metadata.pkl", 'wb') as f:
            import pickle
            pickle.dump({'csv_mtime': os.path.getmtime(csv_path)}, f)
        
        logger.info(f"Created cache with {len(fingerprints):,} fingerprints.")
    
        
    def fit(self, fingerprints: List[np.ndarray], fp_lengths: List[int], smiles: List[str] = None):
        self._log_info(f"Fitting {len(fingerprints):,} fingerprints...")
        self.fingerprints = fingerprints
        self.smiles = smiles
        self.fp_lengths = np.array(fp_lengths)
        self._log_info(f"Fit completed.")
        
    def _save_cache(self):
        """Save the current state to cache."""
        self.fingerprint_cache_dir.mkdir(exist_ok=True)
        fp_array = np.array(self.fingerprints, dtype=object)
        np.save(self.fingerprint_cache_dir / "fingerprints.npy", fp_array)
        if self.smiles:
            smiles_array = np.array(self.smiles, dtype=str)
            np.save(self.fingerprint_cache_dir / "smiles.npy", smiles_array)
    
    def _load_from_cache(self):
        """Load the state from cache."""
        fp_array = np.load(self.fingerprint_cache_dir / "fingerprints.npy", allow_pickle=True)
        self.fingerprints = fp_array.tolist()
        smiles_path = self.fingerprint_cache_dir / "smiles.npy"
        if smiles_path.exists():
            self.smiles = np.load(smiles_path, allow_pickle=True).tolist()


    def search(
        self, 
        query_smiles: np.ndarray, 
        k: int = 5,
        threshold: float = 0.3, 
        batch_size: int = 10000
    ) -> List[Tuple[int, float]]:
        
        query = get_morgan_fp_indices(query_smiles)
        if query is None:
            self._log_info("Invalid query SMILES. Skipping search.")
            return []
        
        heap = []
        n_query = len(query)
        
        comparisons = early_stops = 0
        n_batches = (len(self.fingerprints) + batch_size - 1) // batch_size
        
        with tqdm(total=len(self.fingerprints), disable=not self.verbose, desc="Searching") as pbar:
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(self.fingerprints))
                
                upper_bounds = np.minimum(n_query, self.fp_lengths[start_idx:end_idx]) / \
                             np.maximum(n_query, self.fp_lengths[start_idx:end_idx])
                if len(heap) == k:
                    min_sim = max(heap[0][0], threshold)
                    valid_indices = np.where(upper_bounds >= min_sim)[0] + start_idx
                else:
                    valid_indices = np.where(upper_bounds >= threshold)[0] + start_idx
                
                early_stops += (end_idx - start_idx) - len(valid_indices)
                comparisons += end_idx - start_idx
                
                for idx in valid_indices:
                    similarity = tanimoto_similarity(query, self.fingerprints[idx])
                    
                    if similarity >= threshold:
                        if len(heap) < k:
                            heapq.heappush(heap, (similarity, idx))
                        elif similarity > heap[0][0]:
                            heapq.heapreplace(heap, (similarity, idx))
                
                pbar.update(end_idx - start_idx)
        
        results = [(idx, sim) for sim, idx in sorted(heap, key=lambda x: (-x[0], x[1]))]
        self._log_info(
            "\n" + "="*50 + "\n"
            f"Search Statistics:\n"
            f"\nComparisons: {comparisons:,}"
            f"\nEarly stops: {early_stops:,} ({early_stops/comparisons*100:.1f}%)"
            "\n" + "="*50 + "\n"
        )
        
        return results

