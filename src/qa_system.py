"""
Main QA System combining LSH retrieval with answer generation.
"""

import time
from typing import List, Dict, Tuple, Optional
from src.lsh import LSH, SimHash
from src.baseline import TFIDFRetrieval
from src.data_processing import DocumentProcessor
from src.analytics import QueryPatternMiner, RetrievalAnalytics
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv() # Load variables from .env


class AcademicQASystem:
    """Scalable QA system for academic handbooks."""

    def __init__(self, use_lsh: bool = True, use_simhash: bool = True,
                 use_tfidf: bool = True, use_llm: bool = False):
        """
        Args:
            use_lsh: Enable MinHash + LSH
            use_simhash: Enable SimHash
            use_tfidf: Enable TF-IDF baseline
            use_llm: Use LLM for answer generation
        """
        self.use_lsh = use_lsh
        self.use_simhash = use_simhash
        self.use_tfidf = use_tfidf
        self.use_llm = use_llm

        # Initialize components
        self.processor = DocumentProcessor(chunk_size=300, overlap=50)
        # Use 128 bands × 1 row: P(hit | Jaccard=0.01) ≈ 72%, appropriate for
        # asymmetric query-document retrieval where queries are much shorter than docs.
        self.lsh = LSH(num_hashes=128, num_bands=128) if use_lsh else None
        self.simhash = SimHash(hash_size=64) if use_simhash else None
        self.tfidf = TFIDFRetrieval() if use_tfidf else None

        # Storage
        self.documents = {}  # chunk_id -> chunk_text
        self.doc_metadata = {}  # chunk_id -> {source, page, etc}
        self.simhash_fingerprints = {}  # chunk_id -> fingerprint

        # Analytics
        self.miner = QueryPatternMiner(min_support=0.05)
        self.analytics = RetrievalAnalytics()

        if use_llm:
            self.setup_llm()

    def setup_llm(self):
        """Setup Google Gemini API."""
        api_key = os.getenv('GOOGLE_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            # Using Gemini 3 Flash Preview (Current state-of-the-art in 2026)
            self.model = genai.GenerativeModel('gemini-3-flash-preview')
        else:
            print("Warning: GOOGLE_API_KEY not set. LLM features disabled.")

    def add_document(self, pdf_path: str, doc_id: str):
        """
        Add a handbook PDF to the system.

        Args:
            pdf_path: Path to PDF
            doc_id: Document identifier
        """
        chunks = self.processor.process_handbook(pdf_path, doc_id)

        for chunk_id, (chunk_text, page_num) in chunks.items():
            self.documents[chunk_id] = chunk_text
            self.doc_metadata[chunk_id] = {
                'source': doc_id,
                'page': page_num
            }

            # Index with LSH
            if self.lsh:
                tokens = set(self.processor.tokenize(chunk_text))
                self.lsh.index_document(chunk_id, tokens)

            # Compute SimHash fingerprint
            if self.simhash:
                tokens = self.processor.tokenize(chunk_text)
                self.simhash_fingerprints[chunk_id] = \
                    self.simhash.compute_fingerprint(tokens)

        print(f"Added {len(chunks)} chunks from {doc_id}")

    def fit_baseline(self):
        """Fit TF-IDF baseline on indexed documents."""
        if self.tfidf:
            self.tfidf.fit(self.documents)
            print(f"Fitted TF-IDF on {len(self.documents)} chunks")

    def retrieve_lsh(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Retrieve using MinHash + LSH.

        Returns:
            List of (chunk_id, similarity) tuples
        """
        if not self.lsh:
            return []

        tokens = set(self.processor.tokenize(query))
        # threshold=0.0 — return all candidates from bucket collisions, then rank by
        # estimated Jaccard (which will naturally be low for short queries vs. long docs)
        results = self.lsh.query(tokens, threshold=0.0)
        return results[:top_k]

    def retrieve_simhash(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Retrieve using SimHash.

        Returns:
            List of (chunk_id, similarity) tuples
        """
        if not self.simhash or not self.simhash_fingerprints:
            return []

        tokens = self.processor.tokenize(query)
        # 0.5 threshold: SimHash similarity of 0.5 means ~32 of 64 bits match,
        # which is a loose but useful similarity signal for Q&A retrieval.
        results = self.simhash.query(tokens, self.simhash_fingerprints, threshold=0.5)
        return results[:top_k]

    def retrieve_tfidf(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Retrieve using TF-IDF baseline.

        Returns:
            List of (chunk_id, similarity) tuples
        """
        if not self.tfidf:
            return []

        results = self.tfidf.query(query, top_k=top_k)
        return results

    def retrieve_hybrid(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Retrieve using both LSH and SimHash (Hybrid Ensemble) using Reciprocal Rank Fusion.
        RRF: Score = sum(1 / (k + rank))
        """
        lsh_results = self.retrieve_lsh(query, top_k=top_k * 3)
        simhash_results = self.retrieve_simhash(query, top_k=top_k * 3)

        # RRF constant (standard value is 60)
        k_rrf = 60
        scores = {}
        found_by = {}

        # Process LSH results
        for rank, (chunk_id, sim) in enumerate(lsh_results):
            scores[chunk_id] = scores.get(chunk_id, 0) + (1.0 / (k_rrf + rank + 1))
            found_by[chunk_id] = found_by.get(chunk_id, set()) | {"LSH"}

        # Process SimHash results
        for rank, (chunk_id, sim) in enumerate(simhash_results):
            scores[chunk_id] = scores.get(chunk_id, 0) + (1.0 / (k_rrf + rank + 1))
            found_by[chunk_id] = found_by.get(chunk_id, set()) | {"SIMHASH"}

        # Sort by RRF score
        results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # We need to return similarity for the UI, so we'll store the 'found_by' 
        # in the system temporarily or just return the top_k
        final_results = []
        for chunk_id, rrf_score in results[:top_k]:
            # Use max similarity from original results or a normalized RRF score
            final_results.append((chunk_id, rrf_score))
            
            # Store metadata about which algorithms hit this chunk
            if chunk_id in self.doc_metadata:
                self.doc_metadata[chunk_id]['last_retrieval_methods'] = list(found_by[chunk_id])

        return final_results

    def retrieve(self, query: str, method: str = 'lsh', top_k: int = 10,
                  timings: bool = False) -> Tuple[List[Tuple[str, float]], Optional[Dict]]:
        """
        Main retrieval method.

        Args:
            query: Query text
            method: 'lsh', 'simhash', 'tfidf', or 'hybrid'
            top_k: Number of results
            timings: Return timing information

        Returns:
            (results, timings_dict)
        """
        timing_data = {} if timings else None

        if method == 'lsh':
            start = time.time()
            results = self.retrieve_lsh(query, top_k)
            duration = time.time() - start
        elif method == 'simhash':
            start = time.time()
            results = self.retrieve_simhash(query, top_k)
            duration = time.time() - start
        elif method == 'tfidf':
            start = time.time()
            results = self.retrieve_tfidf(query, top_k)
            duration = time.time() - start
        elif method == 'hybrid':
            start = time.time()
            results = self.retrieve_hybrid(query, top_k)
            duration = time.time() - start
        else:
            raise ValueError(f"Unknown retrieval method: {method}")

        if timing_data is not None:
            timing_data['retrieval_time'] = duration

        chunk_ids = [r[0] for r in results]
        self.analytics.log_performance(method, duration, len(results),
                                       query=query, chunk_ids=chunk_ids)
        # Log query for pattern mining
        self.miner.log_query(query)

        return results, timing_data

    def generate_answer_extractive(self, query: str, retrieved_chunks: List[str]) -> str:
        """
        Generate answer by extracting relevant sentences.

        Args:
            query: Query text
            retrieved_chunks: List of retrieved chunk texts

        Returns:
            Generated answer
        """
        # Find most relevant sentences
        combined_text = " ".join(retrieved_chunks)
        sentences = combined_text.split('. ')

        # Simple relevance ranking based on query word overlap
        query_words = set(self.processor.tokenize(query))
        scored_sentences = []

        for sentence in sentences:
            sent_words = set(self.processor.tokenize(sentence))
            overlap = len(query_words & sent_words)
            if overlap > 0:
                scored_sentences.append((sentence, overlap))

        # Return top sentences
        scored_sentences.sort(key=lambda x: -x[1])
        answer_sentences = [s[0] for s in scored_sentences[:3]]

        return ". ".join(answer_sentences) + "." if answer_sentences else "No relevant information found."

    def generate_answer_llm(self, query: str, retrieved_chunks: List[str], chat_history: List[Dict] = None) -> str:
        """
        Generate answer using Gemini based on retrieved chunks and chat history.

        Args:
            query: Query text
            retrieved_chunks: List of retrieved chunk texts
            chat_history: List of previous messages [{"role": "user/assistant", "content": "..."}]

        Returns:
            Generated answer
        """
        if not hasattr(self, 'model'):
            return self.generate_answer_extractive(query, retrieved_chunks)

        context = "\n".join(retrieved_chunks)
        
        history_str = ""
        if chat_history:
            history_str = "\nConversation History:\n"
            for msg in chat_history[-5:]: # Keep last 5 messages for context
                role = "User" if msg["role"] == "user" else "Assistant"
                history_str += f"{role}: {msg['content']}\n"

        prompt = f"""
        You are an expert academic advisor. Answer the following question based ONLY on the provided context from the university handbook and the conversation history.
        If the answer is not in the context, say "I cannot find the answer in the handbook."
        
        Context:
        {context}
        {history_str}
        Question: {query}
        
        Answer:"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini error: {e}")
            return self.generate_answer_extractive(query, retrieved_chunks)

    def chat(self, query: str, chat_history: List[Dict], method: str = 'lsh', 
             top_k: int = 5, answer_method: str = 'llm') -> Dict:
        """
        Conversational QA method.
        """
        # For chat, we might want to perform retrieval on the query
        # but the LLM should also consider history.
        retrieved, _ = self.retrieve(query, method, top_k, timings=True)
        chunk_ids = [r[0] for r in retrieved]
        chunk_texts = [self.documents[chunk_id] for chunk_id in chunk_ids]

        if answer_method == 'llm' and self.use_llm:
            answer = self.generate_answer_llm(query, chunk_texts, chat_history)
        else:
            answer = self.generate_answer_extractive(query, chunk_texts)

        return {
            'answer': answer,
            'retrieved_chunks': [
                {
                    'id': cid, 
                    'source': self.doc_metadata[cid].get('source', 'Unknown'),
                    'page': self.doc_metadata[cid].get('page', '?')
                } for cid in chunk_ids
            ]
        }

    def answer_query(self, query: str, method: str = 'lsh', top_k: int = 10,
                     answer_method: str = 'extractive') -> Dict:
        """
        Complete QA pipeline: retrieve and generate answer.

        Args:
            query: User query
            method: Retrieval method
            top_k: Number of chunks to retrieve
            answer_method: 'extractive' or 'llm'

        Returns:
            Dict with answer, retrieved chunks, and metadata
        """
        retrieved, timing = self.retrieve(query, method, top_k, timings=True)

        chunk_ids = [r[0] for r in retrieved]
        chunk_texts = [self.documents[chunk_id] for chunk_id in chunk_ids]

        # Generate answer
        if answer_method == 'llm' and self.use_llm:
            answer = self.generate_answer_llm(query, chunk_texts)
        else:
            answer = self.generate_answer_extractive(query, chunk_texts)

        query_tokens = set(self.processor.tokenize(query))
        chunks_result = []
        for chunk_id, similarity in retrieved:
            chunk_tokens = set(self.processor.tokenize(self.documents[chunk_id]))
            # Word coverage: fraction of query keywords present in the chunk
            coverage = len(query_tokens & chunk_tokens) / max(len(query_tokens), 1)
            # Blend coverage (0-1) with normalised similarity for final confidence
            # TF-IDF cosine is already 0-1; Jaccard/SimHash tend to be <0.3 so scale up
            sim_norm = min(similarity * 3.0, 1.0) if method in ('lsh', 'simhash', 'hybrid') else similarity
            confidence = min(1.0, coverage * 0.65 + sim_norm * 0.35)
            chunks_result.append({
                'id': chunk_id,
                'text': self.documents[chunk_id][:200] + "...",
                'source': self.doc_metadata[chunk_id].get('source', 'Unknown'),
                'similarity': similarity,
                'confidence': confidence,
                'found_by': self.doc_metadata[chunk_id].get('last_retrieval_methods', [method.upper()])
            })

        return {
            'query': query,
            'answer': answer,
            'retrieved_chunks': chunks_result,
            'method': method,
            'timing': timing
        }

    def get_statistics(self) -> Dict:
        """Get system statistics and analytics."""
        stats = {
            'total_chunks': len(self.documents),
            'total_tokens': sum(len(self.processor.tokenize(text))
                               for text in self.documents.values()),
            'total_queries': self.analytics.total_query_count(),
            'methods_enabled': {
                'lsh': self.use_lsh,
                'simhash': self.use_simhash,
                'tfidf': self.use_tfidf,
                'llm': self.use_llm
            },
            'performance_summary': self.analytics.get_summary(),
            'hot_topics': self.miner.get_hot_topics(top_n=10),
            'query_history': list(reversed(self.analytics.query_history[-10:])),
            'section_importance': self.analytics.get_section_importance(self.doc_metadata),
        }
        return stats
