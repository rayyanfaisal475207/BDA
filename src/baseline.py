"""
TF-IDF baseline implementation for document retrieval.
Provides exact similarity matching for comparison with LSH methods.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict


class TFIDFRetrieval:
    """TF-IDF based document retrieval (baseline/exact method)."""

    def __init__(self, max_features: int = 5000):
        """
        Args:
            max_features: Maximum number of features for TF-IDF
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True,
            min_df=1,
            max_df=0.95
        )
        self.tfidf_matrix = None
        self.doc_ids = []

    def fit(self, documents: Dict[str, str]):
        """
        Fit TF-IDF model on documents.

        Args:
            documents: Dict of doc_id -> text
        """
        self.doc_ids = list(documents.keys())
        texts = [documents[doc_id] for doc_id in self.doc_ids]

        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Retrieve top-k similar documents.

        Args:
            query_text: Query text
            top_k: Number of results to return

        Returns:
            List of (doc_id, similarity) tuples
        """
        if self.tfidf_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")

        query_vector = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [
            (self.doc_ids[idx], float(similarities[idx]))
            for idx in top_indices
            if similarities[idx] > 0
        ]

        return results

    def get_feature_names(self) -> List[str]:
        """Get TF-IDF feature names."""
        return self.vectorizer.get_feature_names_out().tolist()
