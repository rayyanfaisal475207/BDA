"""
Analytics and pattern mining for the QA system.
Implements Frequent Itemset Mining to identify common query patterns,
and PageRank-style section importance tracking.
"""

import re
import time as time_module
from typing import List, Set, Dict, Tuple
from collections import Counter, defaultdict
import itertools


class QueryPatternMiner:
    """Identify frequent patterns in user queries using Apriori algorithm."""

    def __init__(self, min_support: float = 0.1, stop_words: Set[str] = None):
        """
        Args:
            min_support: Minimum frequency to be considered "frequent"
            stop_words: Set of words to ignore
        """
        self.min_support = min_support
        self.stop_words = stop_words or {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how', 'where',
            'when', 'why', 'who', 'for', 'to', 'in', 'on', 'at', 'by', 'of',
            'with', 'and', 'or', 'if', 'be', 'can', 'do', 'does', 'did'
        }
        self.query_log = []

    def log_query(self, query: str):
        """Add a query to the log for mining."""
        self.query_log.append(query)

    def _get_itemsets(self, query: str) -> Set[str]:
        """Convert query to set of cleaned tokens."""
        words = re.findall(r'\b\w+\b', query.lower())
        return {w for w in words if w not in self.stop_words and len(w) > 2}

    def find_frequent_patterns(self) -> List[Tuple[Tuple[str, ...], int]]:
        """
        Find frequent word patterns in the query log using a simplified Apriori.
        
        Returns:
            List of (pattern_tuple, frequency) sorted by frequency
        """
        if not self.query_log:
            return []

        # Convert logs to transactions
        transactions = [self._get_itemsets(q) for q in self.query_log]
        num_transactions = len(transactions)
        min_count = max(1, int(self.min_support * num_transactions))

        # 1-itemsets
        item_counts = Counter()
        for t in transactions:
            for item in t:
                item_counts[item] += 1
        
        frequent_items = {item for item, count in item_counts.items() if count >= min_count}
        
        # 2-itemsets (patterns of 2 words)
        pair_counts = Counter()
        for t in transactions:
            # Only consider frequent items
            t_frequent = sorted([item for item in t if item in frequent_items])
            if len(t_frequent) >= 2:
                for pair in itertools.combinations(t_frequent, 2):
                    pair_counts[pair] += 1
        
        frequent_patterns = [
            (pair, count) for pair, count in pair_counts.items() if count >= min_count
        ]
        
        # Also include single frequent words as patterns
        for item in frequent_items:
            frequent_patterns.append(((item,), item_counts[item]))

        # Sort by frequency and then by pattern length (longer patterns first)
        return sorted(frequent_patterns, key=lambda x: (-x[1], -len(x[0])))

    def get_hot_topics(self, top_n: int = 5) -> List[Tuple[str, int]]:
        """Identify top N frequently discussed topics."""
        patterns = self.find_frequent_patterns()
        
        # Filter for single-word topics and multi-word phrases
        topics = []
        for pattern, count in patterns:
            topic_str = " ".join(pattern)
            topics.append((topic_str, count))
            
        return topics[:top_n]


class RetrievalAnalytics:
    """Analyze retrieval performance and patterns."""

    def __init__(self):
        self.stats = defaultdict(list)
        self.section_hits = Counter()   # chunk_id -> access count (PageRank proxy)
        self.query_history = []         # last N queries with metadata

    def log_performance(self, method: str, query_time: float, num_results: int,
                        query: str = None, chunk_ids: list = None):
        """Log retrieval performance metrics and track section access."""
        self.stats[method].append({
            'time': query_time,
            'results': num_results
        })
        if chunk_ids:
            for cid in chunk_ids:
                self.section_hits[cid] += 1
        if query:
            self.query_history.append({
                'query': query[:60] + ('…' if len(query) > 60 else ''),
                'method': method.upper(),
                'time_ms': round(query_time * 1000, 3),
                'results': num_results,
            })
            if len(self.query_history) > 50:
                self.query_history = self.query_history[-50:]

    def total_query_count(self) -> int:
        return sum(len(v) for v in self.stats.values())

    def get_section_importance(self, doc_metadata: dict, top_n: int = 8) -> List[Dict]:
        """
        PageRank-style: sections ranked by query access frequency.
        Sections retrieved more often across queries are treated as 'important'.
        """
        results = []
        seen_sources = set()
        for chunk_id, count in self.section_hits.most_common(top_n * 3):
            meta = doc_metadata.get(chunk_id, {})
            source = meta.get('source', 'unknown')
            page = meta.get('page', '?')
            label = f"{source} / p{page}"
            if label not in seen_sources:
                seen_sources.add(label)
                results.append({'label': label, 'hits': count})
            if len(results) >= top_n:
                break
        return results

    def get_summary(self) -> Dict:
        """Get summary of performance metrics."""
        summary = {}
        for method, logs in self.stats.items():
            times = [l['time'] for l in logs]
            results = [l['results'] for l in logs]
            summary[method] = {
                'avg_time': sum(times) / len(times) if times else 0,
                'min_time': min(times) if times else 0,
                'max_time': max(times) if times else 0,
                'avg_results': sum(results) / len(results) if results else 0,
                'total_queries': len(logs)
            }
        return summary
