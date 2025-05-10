from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS

def create_vector_store(documents, embedding_model, save_path):
    """
    Create and save a FAISS vector store from documents.
    
    Args:
        documents: List of LangChain Document objects to embed.
        embedding_model: Name of the Ollama embedding model.
        save_path: Path to save the FAISS index.
    
    Returns:
        None
    """
    try:
        # Initialize embeddings
        embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Create vector store
        vector_store = FAISS.from_documents(documents, embeddings)
        
        # Save vector store
        vector_store.save_local(save_path)
        print(f"Vector store saved to {save_path}")
        
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        raise
