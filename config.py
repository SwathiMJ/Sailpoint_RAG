import os

class Config:
    """
    Configuration class to manage file paths, embeddings, and model settings.
    """

    # --- PDF Configuration ---
    PDF_SOURCE_DIRECTORY: str = "data"
    CHROMA_PERSIST_DIRECTORY: str = "docs/chroma"

    # --- Embedding Model Configuration ---
    EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-large"  # multilingual, good for documents
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 100

    # --- Chat Model Defaults ---
    CHAT_MODEL_NAME: str = "llama-3.1-8b-instant"
    MAX_TOKENS: int = 400
    TEMPERATURE: float = 0.0

    # --- Environment Variables ---
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    def __init__(self):
        # Ensure directories exist
        os.makedirs(self.PDF_SOURCE_DIRECTORY, exist_ok=True)
        os.makedirs(self.CHROMA_PERSIST_DIRECTORY, exist_ok=True)

        print(f"[Config] PDF docs directory: '{self.PDF_SOURCE_DIRECTORY}'")
        print(f"[Config] Chroma DB will be stored at: '{self.CHROMA_PERSIST_DIRECTORY}'")
        if self.GROQ_API_KEY:
            print("[Config] GROQ API key loaded from environment.")
        else:
            print("[Config] Warning: GROQ API key not set!")

# Global instance
config = Config()
