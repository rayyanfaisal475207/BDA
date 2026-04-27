"""Academic Policy QA System using LSH and Big Data techniques."""

__version__ = "1.0.0"
__author__ = "LexiPolicy Dev Team"

from src.lsh import MinHash, LSH, SimHash
from src.baseline import TFIDFRetrieval
from src.qa_system import AcademicQASystem
from src.data_processing import DocumentProcessor
from src.analytics import QueryPatternMiner, RetrievalAnalytics

__all__ = [
    'MinHash',
    'LSH',
    'SimHash',
    'TFIDFRetrieval',
    'AcademicQASystem',
    'DocumentProcessor',
    'QueryPatternMiner',
    'RetrievalAnalytics',
]
