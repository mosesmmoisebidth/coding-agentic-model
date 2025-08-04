# config.py
import os

WORKING_DIR = os.path.abspath("./ai_workspace")

SUPPORTED_FILE_TYPES = [".py", ".md", ".txt", ".json", ".yml", ".yaml", ".csv", ".jsx", ".ts", ".tsx", ".rs", ".java", ".c", ".cc", ".cpp", ".h", ".htmx", ".vb", ".go", ".sh", ".bash", ".zsh",".txt", ".rb"]

AGENT_MODEL = "gpt-4o"

# The OpenAI model to use for creating embeddings for the RAG system.
EMBEDDINGS_MODEL = "text-embedding-3-small"