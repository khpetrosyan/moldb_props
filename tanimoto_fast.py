import os
import heapq
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor

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
    def __init__(self, csv_path, fingerprint_cache_dir=None, verbose=True):
        self.smiles = None
        self._fp_sets = None
        self.verbose = verbose
        self.fp_lengths = None
        self.fingerprints = None
        self.fingerprint_cache_dir = fingerprint_cache_dir or "fingerprint_cache"
        self.searcher = self.load_or_create_cache(csv_path, self.fingerprint_cache_dir)
    
    def _log_info(self, s):
        if self.verbose:
            logger.info(s)
    
    @property
    def fp_sets(self):
        if self._fp_sets is None:
            self._log_info("Creating fingerprint sets on first use...")
            self._fp_sets = [set(fp) for fp in tqdm(self.fingerprints, desc="Creating sets")]
        return self._fp_sets
        
    def fit(self, fingerprints: List[np.ndarray], fp_lengths: List[int], smiles: List[str] = None):
        self._log_info(f"Fitting {len(fingerprints):,} fingerprints...")
        self.fingerprints = fingerprints
        self.smiles = smiles
        self.fp_lengths = np.array(fp_lengths)  # Use pre-computed lengths
        self._fp_sets = None  # Reset cache
        self._log_info(f"Fit completed.")
    
    def save_cache(self, cache_dir: Path):
        cache_dir.mkdir(exist_ok=True)
        fp_array = np.array(self.fingerprints, dtype=object)
        np.save(cache_dir / "fingerprints.npy", fp_array)
        if self.smiles:
            smiles_array = np.array(self.smiles, dtype=str)
            np.save(cache_dir / "smiles.npy", smiles_array)
    
    @classmethod
    def load_cache(cls, cache_dir: Path):
        instance = cls(cache_dir=cache_dir)
        fp_array = np.load(cache_dir / "fingerprints.npy", allow_pickle=True)
        instance.fingerprints = fp_array.tolist()
        smiles_path = cache_dir / "smiles.npy"
        if smiles_path.exists():
            instance.smiles = np.load(smiles_path, allow_pickle=True).tolist()
        instance._fp_sets = None
        
        return instance

    @staticmethod
    def load_or_create_cache(csv_file: str, fingerprint_cache_dir: str):
        CACHE_DIR = Path(fingerprint_cache_dir)
        FP_CACHE = CACHE_DIR / "fingerprints.npy"
        METADATA_CACHE = CACHE_DIR / "metadata.pkl"

        CACHE_DIR.mkdir(exist_ok=True)
        csv_mtime = os.path.getmtime(csv_file)
        cache_exists = all(p.exists() for p in [FP_CACHE, METADATA_CACHE])
        
        if cache_exists:
            with open(METADATA_CACHE, 'rb') as f:
                try:
                    import pickle
                    metadata = pickle.load(f)
                    if metadata.get('csv_mtime') == csv_mtime:
                        searcher = FastSearch.load_cache(fingerprint_cache_dir)
                        df = pd.read_csv(csv_file)
                        searcher.fp_lengths = df['Fingerprint_Sum'].values
                        logger.info(f"Loaded {len(searcher.fingerprints):,} fingerprints from cache")
                        return searcher
                except:
                    pass
        logger.info("Cache not found or outdated. Creating new cache...")        
        
        df = pd.read_csv(csv_file)        
        n_cores = multiprocessing.cpu_count()
        chunk_size = max(1, len(df) // (n_cores - 2)) # Leave 2 cores for other tasks
        chunks = [df['Morgan_Fingerprint'][i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        
        fingerprints = []
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            futures = [executor.submit(batch_parse_fingerprints, chunk) for chunk in chunks]
            for future in tqdm(futures, desc="Processing chunks"):
                batch_fps = future.result()
                fingerprints.extend(batch_fps)
        
        smiles = df['SMILES'].tolist()
        fp_lengths = df['Fingerprint_Sum'].tolist()  # Use pre-computed sums
        
        searcher = FastSearch()
        searcher.fit(fingerprints, fp_lengths, smiles)
        searcher.save_cache(CACHE_DIR)
        with open(METADATA_CACHE, 'wb') as f:
            import pickle
            pickle.dump({'csv_mtime': csv_mtime}, f)
        logger.info(f"Created cache with {len(fingerprints):,} fingerprints.")
                 
        return searcher


    def search(
        self, 
        query: np.ndarray, 
        k: int = 5,
        threshold: float = 0.3, 
        batch_size: int = 10000
    ) -> List[Tuple[int, float]]:
        
        heap = []
        n_query = len(query)
        query_set = set(query)
        
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
                    intersection = len(query_set.intersection(self.fp_sets[idx]))
                    union = n_query + self.fp_lengths[idx] - intersection
                    similarity = intersection / union if union > 0 else 0.0
                    
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



