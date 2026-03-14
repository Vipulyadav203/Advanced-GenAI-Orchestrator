import logging
import os
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
from langchain_openai import OpenAIEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Manages the knowledge base using ChromaDB.
    Follows SOLID: Interface Segregation and Single Responsibility.
    """
    def __init__(self, collection_name: str = "knowledge_base"):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        logger.info(f"Initialized VectorStoreManager with collection: {collection_name}")

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """
        Adds multiple documents to the vector store.
        """
        logger.info(f"Adding {len(documents)} documents to vector store.")
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info("Successfully added documents to vector store.")
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise

    def similarity_search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Performs a similarity search for the given query.
        """
        logger.info(f"Performing similarity search for query: {query}")
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            logger.info("Similarity search completed.")
            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return {"documents": [], "metadatas": [], "distances": []}
