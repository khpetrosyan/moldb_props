import numpy as np
import heapq
import pandas as pd
from typing import List, Tuple, Optional
from tqdm import tqdm
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import pickle
import os
from pathlib import Path

CACHE_DIR = Path("fingerprint_cache")
FP_CACHE = CACHE_DIR / "fingerprints.pkl"
METADATA_CACHE = CACHE_DIR / "metadata.pkl"

def parse_fingerprint(fp_str: str) -> Optional[np.ndarray]:
    """Convert comma-separated string of indices to numpy array. Returns None for invalid inputs."""
    if pd.isna(fp_str) or not isinstance(fp_str, str):
        return None
    try:
        # Debug print
        arr = np.fromstring(fp_str.strip('"'), sep=',', dtype=np.int32)
        return arr
    except (ValueError, AttributeError) as e:
        print("Parse error:", str(e))
        return None

def batch_parse_fingerprints(chunk):
    """Parse a batch of fingerprints in parallel."""
    return [parse_fingerprint(fp) for fp in chunk if not pd.isna(fp)]

class FastSearch:
    def __init__(self):
        self.fingerprints = None
        self.smiles = None
        self.fp_lengths = None
        self.fp_sets = None  # Cache for set representations
        
    def fit(self, fingerprints: List[np.ndarray], smiles: List[str] = None):
        print(f"Fitting {len(fingerprints):,} fingerprints...")
        start = time.time()
        self.fingerprints = fingerprints
        self.smiles = smiles
        self.fp_lengths = np.array([len(fp) for fp in fingerprints])
        # Precompute sets for faster intersection operations
        print("Precomputing fingerprint sets...")
        self.fp_sets = [set(fp) for fp in tqdm(fingerprints, desc="Creating sets")]
        print(f"Fit completed in {time.time() - start:.2f} seconds")
    
    def save_cache(self, cache_file: Path):
        """Save the preprocessed data to cache."""
        cache_data = {
            'fingerprints': self.fingerprints,
            'smiles': self.smiles,
            'fp_lengths': self.fp_lengths,
            'fp_sets': self.fp_sets
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    
    @classmethod
    def load_cache(cls, cache_file: Path):
        """Load preprocessed data from cache."""
        instance = cls()
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        instance.fingerprints = cache_data['fingerprints']
        instance.smiles = cache_data['smiles']
        instance.fp_lengths = cache_data['fp_lengths']
        instance.fp_sets = cache_data['fp_sets']
        return instance
    
    def search(self, query: np.ndarray, k: int = 5, show_progress: bool = True, batch_size: int = 10000) -> List[Tuple[int, float]]:
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
                
                # Quick upper bound check using precomputed lengths
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
                    # Use precomputed sets for intersection
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
    cache_exists = FP_CACHE.exists() and METADATA_CACHE.exists()
    
    if cache_exists:
        with open(METADATA_CACHE, 'rb') as f:
            metadata = pickle.load(f)
            if metadata.get('csv_mtime') == csv_mtime:
                print("Loading from cache...")
                start = time.time()
                searcher = FastSearch.load_cache(FP_CACHE)
                load_time = time.time() - start
                print(f"Loaded {len(searcher.fingerprints):,} fingerprints from cache in {load_time:.2f} seconds")
                return searcher, load_time
    
    # Cache doesn't exist or is outdated - create new
    print("Cache not found or outdated. Creating new cache...")
    start = time.time()
    
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df):,} molecules")
    
    n_cores = multiprocessing.cpu_count()
    chunk_size = len(df) // n_cores
    chunks = [df['Morgan_Fingerprint'][i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    fingerprints = []
    invalid_count = 0
    
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
    searcher.save_cache(FP_CACHE)
    with open(METADATA_CACHE, 'wb') as f:
        pickle.dump({'csv_mtime': csv_mtime}, f)
    
    process_time = time.time() - start
    print(f"Created cache with {len(fingerprints):,} fingerprints in {process_time:.2f} seconds")
    print(f"Invalid/NaN fingerprints found: {invalid_count:,}")
    
    return searcher, process_time
