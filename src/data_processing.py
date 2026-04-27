"""
Data ingestion and preprocessing for handbook documents.
Handles PDF/text conversion and chunking.
"""

import re
import os
from typing import List, Dict, Tuple
import pdfplumber
from pathlib import Path


class DocumentProcessor:
    """Process and chunk documents for the QA system."""

    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        """
        Args:
            chunk_size: Target chunk size in words
            overlap: Overlap between chunks in words
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text
        """
        text = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
        except Exception as e:
            print(f"Error extracting PDF {pdf_path}: {e}")
            return ""

        return "\n".join(text)

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\:\;\-\(\)\?]', '', text)

        # Remove common PDF artifacts
        text = re.sub(r'Page \d+', '', text)
        text = re.sub(r'\|\s+\|', '', text)

        return text.strip()

    def chunk_text(self, text: str, doc_id: str) -> List[Tuple[str, str, int]]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            doc_id: Document identifier (for reference)

        Returns:
            List of (chunk_id, chunk_text, page_number) tuples
        """
        words = text.split()
        chunks = []
        chunk_num = 0

        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]

            if len(chunk_words) > 10:  # Skip very small chunks
                chunk_text = " ".join(chunk_words)
                chunk_id = f"{doc_id}_chunk_{chunk_num}"
                chunks.append((chunk_id, chunk_text, chunk_num))
                chunk_num += 1

        return chunks

    def process_handbook(self, pdf_path: str, doc_id: str) -> Dict[str, Tuple[str, int]]:
        """
        Process a handbook PDF into chunks.

        Args:
            pdf_path: Path to PDF
            doc_id: Document identifier

        Returns:
            Dict of chunk_id -> (chunk_text, chunk_number)
        """
        # Extract and clean text
        raw_text = self.extract_text_from_pdf(pdf_path)
        clean_text = self.clean_text(raw_text)

        # Chunk the text
        chunks = self.chunk_text(clean_text, doc_id)

        return {
            chunk_id: (chunk_text, page_num)
            for chunk_id, chunk_text, page_num in chunks
        }

    # Common English stop words to remove before MinHash/SimHash comparison
    _STOP_WORDS = frozenset({
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
        'on', 'with', 'at', 'by', 'from', 'as', 'or', 'and', 'but', 'not',
        'this', 'that', 'these', 'those', 'it', 'its', 'if', 'so', 'up',
        'out', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'than', 'then', 'there', 'when', 'where', 'who',
        'which', 'what', 'how', 'no', 'nor', 'only', 'own', 'same', 'too',
        'very', 'just', 'about', 'above', 'after', 'also', 'into', 'through',
        'during', 'before', 'between', 'you', 'he', 'she', 'we', 'they',
        'their', 'your', 'his', 'her', 'our', 'my', 'i', 'me', 'him', 'us',
    })

    def tokenize(self, text: str, remove_stop_words: bool = True) -> List[str]:
        """
        Tokenize text into meaningful content words.

        Args:
            text: Text to tokenize
            remove_stop_words: Filter common English stop words (default True)

        Returns:
            List of tokens (min length 2, stop-words removed when requested)
        """
        text = text.lower()
        tokens = re.findall(r'\b[a-z]\w+\b', text)  # alphabetic words only
        if remove_stop_words:
            tokens = [t for t in tokens if t not in self._STOP_WORDS and len(t) >= 2]
        return tokens

    @staticmethod
    def download_handbooks(save_dir: str = "data/handbooks"):
        """
        Download handbooks from NUST website (if available).

        Args:
            save_dir: Directory to save PDFs
        """
        os.makedirs(save_dir, exist_ok=True)

        # Handbook URLs (these are examples - adjust based on actual URLs)
        urls = {
            "UG_Handbook": "https://seecs.nust.edu.pk/downloads/student-handbooks/UG_Handbook.pdf",
            "PG_Handbook": "https://seecs.nust.edu.pk/downloads/student-handbooks/PG_Handbook.pdf"
        }

        import requests
        from tqdm import tqdm

        for name, url in urls.items():
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    filepath = os.path.join(save_dir, f"{name}.pdf")
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded {name} to {filepath}")
                else:
                    print(f"Failed to download {name}: {response.status_code}")
            except Exception as e:
                print(f"Error downloading {name}: {e}")
