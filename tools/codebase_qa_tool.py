from langchain.tools import BaseTool
from services.vectorstore_service import VectorStoreService
from typing import Type
from pydantic import BaseModel, Field

class CodebaseQAToolInput(BaseModel):
    query: str = Field(description="A clear and specific question about the codebase.")

class CodebaseQATool(BaseTool):
    """A tool to ask questions about the codebase and get contextually relevant code snippets."""
    name: str = "codebase_qa_tool"
    description: str = (
        "Use this tool to ask questions about the current codebase. "
        "It performs a semantic search over the files in the workspace "
        "and returns the most relevant code snippets. "
        "Input should be a clear, specific question about the code."
    )
    args_schema: Type[BaseModel] = CodebaseQAToolInput
    vectorstore_service: VectorStoreService

    def _run(self, query: str) -> str:
        retriever = self.vectorstore_service.get_retriever()
        if not retriever:
            return "Vector store not available. The workspace might be empty or has not been indexed yet. Try using the 'ls' command to see files first."

        try:
            results = retriever.invoke(query)
            if not results:
                return "I couldn't find any relevant code snippets for your question."
            
            # Format the results for the LLM
            context = "\n---\n".join([doc.page_content for doc in results])
            return f"Here are the most relevant code snippets I found:\n\n{context}"
        except Exception as e:
            return f"An error occurred while searching the codebase: {e}"