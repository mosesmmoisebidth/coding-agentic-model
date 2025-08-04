# main.py
import os
import config
from agentic import create_agent_executor
from services.vectorstore_service import VectorStoreService

def main():
    # Ensure the OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("🔴 Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it before running the agent.")
        return
    if not os.environ.get("TAVILY_API_KEY"): # <-- NEW CHECK
        print("🔴 Error: TAVILY_API_KEY environment variable not set.")
        print("Please get a key from tavily.com and set it.")
        return
    # Create the working directory if it doesn't exist
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    print(f"✅ Agent working directory set to: {config.WORKING_DIR}")
    # Initialize our RAG service
    vectorstore_service = VectorStoreService(
        working_dir=config.WORKING_DIR,
        supported_file_types=config.SUPPORTED_FILE_TYPES,
        embeddings_model=config.EMBEDDINGS_MODEL
    )
    # Create the agent executor
    agent_executor = create_agent_executor(vectorstore_service)
    print("\n🤖 AI Developer Agent is ready! (with Collaboration 🤝)")
    print("I may ask for clarification if a request is ambiguous.") # <-- NEW LINE
    print("Type 'reindex' to update my knowledge of the workspace.")
    print("Type 'exit' or 'quit' to end the session.")
    
    chat_history = []

    while True:
        user_input = input("\n🗣️  You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting.")
            break
        
        # Add a special command to trigger re-indexing
        if user_input.lower() == "reindex":
            print("🔄 Re-indexing workspace...")
            vectorstore_service.reindex()
            print("✅ Workspace re-indexed successfully.")
            continue

        try:
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            # Add interaction to chat history
            chat_history.extend([
                ("human", user_input),
                ("ai", response["output"])
            ])
            
            print(f"\n🤖 Agent:\n{response['output']}")
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()