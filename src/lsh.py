"""
Locality Sensitive Hashing (LSH) implementation for document similarity.
Includes MinHash + LSH and SimHash implementations.
"""

import hashlib
import numpy as np
from typing import List, Set, Tuple, Dict
from collections import defaultdict


class MinHash:
    """MinHash signature for approximate set similarity."""

    def __init__(self, num_hashes: int = 128):
        """
        Args:
            num_hashes: Number of hash functions to use
        """
        self.num_hashes = num_hashes
        self.hash_seeds = np.random.randint(0, 2**32, num_hashes)

    def _hash_token(self, token: str, seed: int) -> int:
        """Hash a single token with a seed."""
        h = hashlib.md5(f"{token}_{seed}".encode()).hexdigest()
        return int(h, 16)

    def compute_signature(self, tokens: Set[str]) -> np.ndarray:
        """
        Compute MinHash signature for a set of tokens.

        Args:
            tokens: Set of tokens/words

        Returns:
            Array of hash values (signature)
        """
        signature = np.full(self.num_hashes, float('inf'))

        for token in tokens:
            for i, seed in enumerate(self.hash_seeds):
                hash_val = self._hash_token(token, seed)
                signature[i] = min(signature[i], hash_val)

        return signature

    def jaccard_similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """
        Estimate Jaccard similarity from signatures.

        Args:
            sig1, sig2: MinHash signatures

        Returns:
            Estimated Jaccard similarity [0, 1]
        """
        matches = np.sum(sig1 == sig2)
        return matches / len(sig1)


class LSH:
    """Locality Sensitive Hashing for efficient similarity search."""

    def __init__(self, num_hashes: int = 128, num_bands: int = 8):
        """
        Args:
            num_hashes: Number of MinHash functions
            num_bands: Number of bands (rows per band = num_hashes / num_bands)
        """
        self.minhash = MinHash(num_hashes)
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.buckets = [defaultdict(list) for _ in range(num_bands)]
        self.signatures = {}

    def index_document(self, doc_id: str, tokens: Set[str]):
        """
        Index a document by computing MinHash and adding to buckets.

        Args:
            doc_id: Unique document identifier
            tokens: Set of tokens in the document
        """
        signature = self.minhash.compute_signature(tokens)
        self.signatures[doc_id] = signature

        # Hash each band
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_sig = signature[start:end]

            # Create bucket key from band signature
            bucket_key = tuple(band_sig)
            self.buckets[band_idx][bucket_key].append(doc_id)

    def query(self, tokens: Set[str], threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Find similar documents using LSH.

        Args:
            tokens: Query tokens
            threshold: Minimum Jaccard similarity

        Returns:
            List of (doc_id, similarity) tuples
        """
        query_sig = self.minhash.compute_signature(tokens)
        candidates = set()

        # Collect candidates from matching buckets
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_sig = query_sig[start:end]
            bucket_key = tuple(band_sig)

            if bucket_key in self.buckets[band_idx]:
                candidates.update(self.buckets[band_idx][bucket_key])

        # Compute exact similarity for candidates
        results = []
        for doc_id in candidates:
            similarity = self.minhash.jaccard_similarity(
                query_sig, self.signatures[doc_id]
            )
            if similarity >= threshold:
                results.append((doc_id, similarity))

        return sorted(results, key=lambda x: -x[1])


class SimHash:
    """SimHash for approximate document similarity using fingerprints."""

    def __init__(self, hash_size: int = 64):
        """
        Args:
            hash_size: Size of fingerprint in bits
        """
        self.hash_size = hash_size

    def _hash_token(self, token: str) -> np.ndarray:
        """Hash token to binary vector."""
        h = hashlib.md5(token.encode()).hexdigest()
        binary = bin(int(h, 16))[2:].zfill(self.hash_size)
        return np.array([int(b) for b in binary])

    def compute_fingerprint(self, tokens: List[str]) -> np.ndarray:
        """
        Compute SimHash fingerprint for document.

        Args:
            tokens: List of tokens

        Returns:
            Binary fingerprint array
        """
        v = np.zeros(self.hash_size, dtype=np.int32)

        for token in tokens:
            h = self._hash_token(token)
            v += np.where(h == 1, 1, -1)

        # Convert to binary: 1 if positive, 0 if negative
        fingerprint = np.where(v > 0, 1, 0)
        return fingerprint

    def hamming_distance(self, fp1: np.ndarray, fp2: np.ndarray) -> int:
        """Compute Hamming distance between two fingerprints."""
        return np.sum(fp1 != fp2)

    def similarity(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """
        Compute similarity from Hamming distance.

        Returns:
            Similarity in [0, 1]
        """
        dist = self.hamming_distance(fp1, fp2)
        return 1 - (dist / self.hash_size)

    def query(self, tokens: List[str], fingerprints: Dict[str, np.ndarray],
              threshold: float = 0.8) -> List[Tuple[str, float]]:
        """
        Find similar documents.

        Args:
            tokens: Query tokens
            fingerprints: Dict of doc_id -> fingerprint
            threshold: Minimum similarity (based on Hamming distance)

        Returns:
            List of (doc_id, similarity) tuples
        """
        query_fp = self.compute_fingerprint(tokens)
        results = []

        for doc_id, fp in fingerprints.items():
            sim = self.similarity(query_fp, fp)
            if sim >= threshold:
                results.append((doc_id, sim))

        return sorted(results, key=lambda x: -x[1])
