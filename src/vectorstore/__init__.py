"""Vector store package."""

from .faiss_store import (
    FAISSVectorStore,
    VectorStoreDimensionError,
    VectorStoreError,
    VectorStorePersistenceError,
)

__all__ = [
    "FAISSVectorStore",
    "VectorStoreError",
    "VectorStoreDimensionError",
    "VectorStorePersistenceError",
]