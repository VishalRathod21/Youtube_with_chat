from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_embeddings():
    """
    Get the embedding model instance.
    This is an alias for get_embedding_model() for backward compatibility.
    """
    return get_embedding_model()