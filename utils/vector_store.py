from typing import List, Any
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStore

def create_vector_store(texts: List[str], embeddings: Embeddings) -> VectorStore:
    """
    Create a vector store from a list of text chunks.
    
    Args:
        texts: List of text chunks to be vectorized
        embeddings: Embedding model to use
        
    Returns:
        VectorStore: Created vector store
    """
    try:
        return FAISS.from_texts(texts=texts, embedding=embeddings)
    except Exception as e:
        raise Exception(f"Error creating vector store: {str(e)}")

def get_similar_docs(vector_store: VectorStore, query: str, k: int = 3) -> List[Any]:
    """
    Get similar documents from the vector store.
    
    Args:
        vector_store: Vector store to search in
        query: Query string
        k: Number of similar documents to retrieve
        
    Returns:
        List of similar documents
    """
    try:
        return vector_store.similarity_search(query=query, k=k)
    except Exception as e:
        raise Exception(f"Error searching vector store: {str(e)}")
