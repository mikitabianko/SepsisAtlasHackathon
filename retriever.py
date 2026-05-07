import os
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever


class ManualEnsembleRetriever:
    """Simple hybrid retriever merging BM25 and vector search results."""

    def __init__(self, bm25_retriever, vector_retriever):
        self.retrievers = [bm25_retriever, vector_retriever]

    def invoke(self, query: str) -> list:
        bm25_results = self.retrievers[0].invoke(query)
        vector_results = self.retrievers[1].invoke(query)

        # Merge and deduplicate by content
        seen = set()
        merged = []
        for doc in bm25_results + vector_results:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged.append(doc)
        return merged


class SepsisRetriever:
    """Hybrid retriever combining dense (Chroma) and sparse (BM25) search."""

    def __init__(self, persist_directory="./sepsis_db"):
        # Load local embeddings model optimized for scientific English
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.persist_directory = persist_directory
        self.ensemble_retriever = None

    def ingest(self, chunks: list):
        """
        Process and vectorize text chunks into the database.
        Expected format: [{"text": "...", "metadata": {"filename": "...", "page": 1}}]
        """
        if not chunks:
            print("Warning: Empty chunks list. Nothing to ingest.")
            return

        print(f"Ingesting {len(chunks)} chunks...")

        # Bind text and metadata for traceability
        docs = [Document(page_content=c["text"], metadata=c["metadata"]) for c in chunks]

        # Initialize/update Chroma vector database
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

        # Setup retrievers for hybrid search
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 5

        # Combine dense and sparse retrievers (40% keyword, 60% semantic)
        self.ensemble_retriever = ManualEnsembleRetriever(bm25_retriever, vector_retriever)
        print("Database successfully built and ready for search.")

    def retrieve(self, query: str, top_k: int = 5) -> list:
        """
        Search for top_k relevant chunks based on the query.
        Returns a list of dicts: [{"text": "...", "metadata": {...}}]
        """
        if not self.ensemble_retriever:
            raise ValueError("Database is empty or not initialized. Call ingest() first.")

        # Update retrieval limit dynamically
        self.ensemble_retriever.retrievers[0].k = top_k
        self.ensemble_retriever.retrievers[1].search_kwargs["k"] = top_k

        # Execute hybrid search
        results = self.ensemble_retriever.invoke(query)

        # Format output for the LLM module
        final_output = []
        for doc in results[:top_k]:
            final_output.append({
                "text": doc.page_content,
                "metadata": doc.metadata
            })

        return final_output


# ==========================================
# TESTING BLOCK
# ==========================================
if __name__ == "__main__":
    # Mock data from the parsing module
    mock_chunks = [
        {"text": "The initial lactate level was a strong predictor of 28-day mortality (OR 1.5).", "metadata": {"filename": "Leona_2025.pdf", "page": 3}},
        {"text": "We used SOFA score to assess organ failure on day 1. Blood pressure was normal.", "metadata": {"filename": "Leona_2025.pdf", "page": 4}},
        {"text": "Sample size included 540 patients with septic shock, average age was 65.", "metadata": {"filename": "Smith_2024.pdf", "page": 2}}
    ]

    # Initialize and load data
    rag_system = SepsisRetriever()
    rag_system.ingest(mock_chunks)

    # Test search
    user_query = "What is the relationship between initial lactate and mortality?"
    retrieved_data = rag_system.retrieve(user_query, top_k=2)

    print("\n--- SEARCH RESULTS ---")
    for idx, item in enumerate(retrieved_data, 1):
        print(f"\n[{idx}] File: {item['metadata']['filename']} (Page: {item['metadata']['page']})")
        print(f"Text: {item['text']}")