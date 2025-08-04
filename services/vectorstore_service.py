from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

class VectorStoreService:
    """
    Manages the creation and retrieval of the vector store for the codebase.
    """
    def __init__(self, working_dir: str, supported_file_types: list, embeddings_model: str):
        self.working_dir = working_dir
        self.supported_file_types = supported_file_types
        self.embeddings = OpenAIEmbeddings(model=embeddings_model)
        self.vector_store = None
        # Use a robust splitter for code
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        print("VectorStoreService initialized.")

    def _load_documents(self):
        """Loads all supported files from the working directory."""
        print("Loading documents from workspace...")
        loaders = []
        for file_type in self.supported_file_types:
            # Use glob to find all files with the supported extension
            loader = DirectoryLoader(
                self.working_dir,
                glob=f"**/*{file_type}",
                loader_cls=TextLoader,
                show_progress=False,
                use_multithreading=True,
                silent_errors=True # Ignore files that can't be read
            )
            loaders.append(loader)

        docs = []
        for loader in loaders:
            try:
                loaded_docs = loader.load()
                if loaded_docs:
                    docs.extend(loaded_docs)
            except Exception as e:
                print(f"Error loading files with a loader: {e}")
        
        if not docs:
            print("No documents found to load.")
        else:
            print(f"Loaded {len(docs)} documents.")
        return docs

    def reindex(self):
        """Re-indexes all files in the workspace and creates the vector store."""
        documents = self._load_documents()
        if not documents:
            self.vector_store = None
            print("Workspace is empty. No vector store created.")
            return

        print("Splitting and embedding documents...")
        texts = self.text_splitter.split_documents(documents)
        
        if not texts:
            print("No text chunks to index after splitting.")
            self.vector_store = None
            return

        # Create the FAISS vector store from the texts
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
        print(f"Vector store created successfully with {len(texts)} text chunks.")

    def get_retriever(self, k: int = 5):
        """
        Returns a retriever for the vector store.
        If the store doesn't exist, it returns None.
        """
        if self.vector_store:
            return self.vector_store.as_retriever(search_kwargs={"k": k})
        return None