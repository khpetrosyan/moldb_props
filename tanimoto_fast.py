import numpy as np
import heapq
import pandas as pd
from typing import List, Tuple, Optional
from tqdm import tqdm
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path

CACHE_DIR = Path("fingerprint_cache")
FP_CACHE = CACHE_DIR / "fingerprints.npy"
SMILES_CACHE = CACHE_DIR / "smiles.npy"
LENGTHS_CACHE = CACHE_DIR / "lengths.npy"
METADATA_CACHE = CACHE_DIR / "metadata.pkl"

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
    def __init__(self):
        self.fingerprints = None
        self.smiles = None
        self.fp_lengths = None
        self._fp_sets = None
    
    @property
    def fp_sets(self):
        # Lazy initialization of sets
        if self._fp_sets is None:
            print("Creating fingerprint sets on first use...")
            self._fp_sets = [set(fp) for fp in tqdm(self.fingerprints, desc="Creating sets")]
        return self._fp_sets
        
    def fit(self, fingerprints: List[np.ndarray], smiles: List[str] = None):
        print(f"Fitting {len(fingerprints):,} fingerprints...")
        start = time.time()
        self.fingerprints = fingerprints
        self.smiles = smiles
        self.fp_lengths = np.array([len(fp) for fp in fingerprints])
        self._fp_sets = None  # Reset cache
        print(f"Fit completed in {time.time() - start:.2f} seconds")
    
    def save_cache(self, cache_dir: Path):
        """Save the preprocessed data to cache using memory-mapped files."""
        cache_dir.mkdir(exist_ok=True)
        
        # Save fingerprints as a structured numpy array
        fp_array = np.array(self.fingerprints, dtype=object)
        np.save(cache_dir / "fingerprints.npy", fp_array)
        
        # Save SMILES as numpy array
        if self.smiles:
            smiles_array = np.array(self.smiles, dtype=str)
            np.save(cache_dir / "smiles.npy", smiles_array)
        
        # Save lengths as numpy array
        np.save(cache_dir / "lengths.npy", self.fp_lengths)
    
    @classmethod
    def load_cache(cls, cache_dir: Path):
        """Load preprocessed data from cache using memory mapping where possible."""
        instance = cls()
        
        # Load fingerprints
        fp_array = np.load(cache_dir / "fingerprints.npy", allow_pickle=True)
        instance.fingerprints = fp_array.tolist()
        
        # Load SMILES if exists
        smiles_path = cache_dir / "smiles.npy"
        if smiles_path.exists():
            instance.smiles = np.load(smiles_path, allow_pickle=True).tolist()
        
        # Load lengths using memory mapping
        instance.fp_lengths = np.load(cache_dir / "lengths.npy", mmap_mode='r')
        instance._fp_sets = None
        
        return instance
    
    def search(self, query: np.ndarray, k: int = 5, show_progress: bool = True, batch_size: int = 10000) -> List[Tuple[int, float]]:
        # Search implementation remains the same
        start = time.time()
        n_query = len(query)
        heap = []
        query_set = set(query)
        
        comparisons = early_stops = 0
        n_batches = (len(self.fingerprints) + batch_size - 1) // batch_size
        
        with tqdm(total=len(self.fingerprints), disable=not show_progress, desc="Searching") as pbar:
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(self.fingerprints))
                
                upper_bounds = np.minimum(n_query, self.fp_lengths[start_idx:end_idx]) / \
                             np.maximum(n_query, self.fp_lengths[start_idx:end_idx])
                
                if len(heap) == k:
                    min_sim = heap[0][0]
                    valid_indices = np.where(upper_bounds >= min_sim)[0] + start_idx
                else:
                    valid_indices = np.arange(start_idx, end_idx)
                
                early_stops += (end_idx - start_idx) - len(valid_indices)
                comparisons += end_idx - start_idx
                
                for idx in valid_indices:
                    intersection = len(query_set.intersection(self.fp_sets[idx]))
                    union = n_query + self.fp_lengths[idx] - intersection
                    similarity = intersection / union if union > 0 else 0.0
                    
                    if len(heap) < k:
                        heapq.heappush(heap, (similarity, idx))
                    elif similarity > heap[0][0]:
                        heapq.heapreplace(heap, (similarity, idx))
                
                pbar.update(end_idx - start_idx)
        
        results = [(idx, sim) for sim, idx in sorted(heap, key=lambda x: (-x[0], x[1]))]
        
        search_time = time.time() - start
        fps = len(self.fingerprints) / search_time if search_time > 0 else 0
        
        print(f"\nSearch Statistics:")
        print(f"Total time: {search_time:.2f} seconds")
        print(f"Speed: {fps:,.0f} fingerprints/second")
        print(f"Comparisons: {comparisons:,}")
        print(f"Early stops: {early_stops:,} ({early_stops/comparisons*100:.1f}%)")
        
        return results

def load_or_create_cache(csv_file: str) -> Tuple[FastSearch, float]:
    """Load fingerprints from cache or create if not exists."""
    CACHE_DIR.mkdir(exist_ok=True)
    
    # Check if cached data exists and is newer than CSV
    csv_mtime = os.path.getmtime(csv_file)
    cache_exists = all(p.exists() for p in [FP_CACHE, LENGTHS_CACHE])
    
    if cache_exists:
        with open(METADATA_CACHE, 'rb') as f:
            try:
                import pickle
                metadata = pickle.load(f)
                if metadata.get('csv_mtime') == csv_mtime:
                    print("Loading from cache...")
                    start = time.time()
                    searcher = FastSearch.load_cache(CACHE_DIR)
                    load_time = time.time() - start
                    print(f"Loaded {len(searcher.fingerprints):,} fingerprints from cache in {load_time:.2f} seconds")
                    return searcher, load_time
            except:
                pass
    
    # Cache doesn't exist or is outdated - create new
    print("Cache not found or outdated. Creating new cache...")
    start = time.time()
    
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df):,} molecules")
    
    n_cores = multiprocessing.cpu_count()
    chunk_size = len(df) // n_cores
    chunks = [df['Morgan_Fingerprint'][i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    fingerprints = []
    
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = [executor.submit(batch_parse_fingerprints, chunk) for chunk in chunks]
        for future in tqdm(futures, desc="Processing chunks"):
            batch_fps = future.result()
            fingerprints.extend(batch_fps)
    
    smiles = df['SMILES'].tolist()
    invalid_count = len(df) - len(fingerprints)
    
    searcher = FastSearch()
    searcher.fit(fingerprints, smiles)
    
    # Save to cache
    searcher.save_cache(CACHE_DIR)
    with open(METADATA_CACHE, 'wb') as f:
        import pickle
        pickle.dump({'csv_mtime': csv_mtime}, f)
    
    process_time = time.time() - start
    print(f"Created cache with {len(fingerprints):,} fingerprints in {process_time:.2f} seconds")
    print(f"Invalid/NaN fingerprints found: {invalid_count:,}")
    
    return searcher, process_time