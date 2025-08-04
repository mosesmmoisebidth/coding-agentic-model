# main.py
import os
import config
from agent_creator import create_agent_executor # <-- CORRECTED IMPORT
from ui import UI
from services.vectorstore_service import VectorStoreService

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from run_team import app as team_app

# --- Router Function (remains the same) ---
def create_router_chain():
    router_prompt_template = """
You are an expert at analyzing user requests for a software development AI agent.
Your task is to classify the user's request into one of two categories: 'single_agent' or 'ai_team'.
Respond with ONLY the category name and nothing else.
Examples for 'single_agent': "List files", "What is the capital of France?", "Write 'hello' to a file.", "What is the latest version of Flask?"
Examples for 'ai_team': "Build a blog app", "Create a FastAPI endpoint and test it.", "Refactor the database module."
User Request: "{input}"
"""
    llm = ChatOpenAI(model=config.AGENT_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_template(router_prompt_template)
    return prompt | llm | StrOutputParser()

def main():
    # --- Setup ---
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
        print("🔴 Error: Make sure both OPENAI_API_KEY and TAVILY_API_KEY environment variables are set.")
        return

    os.makedirs(config.WORKING_DIR, exist_ok=True)
    ui = UI()

    vectorstore_service = VectorStoreService(
        working_dir=config.WORKING_DIR,
        supported_file_types=config.SUPPORTED_FILE_TYPES,
        embeddings_model=config.EMBEDDINGS_MODEL
    )
    single_agent_executor = create_agent_executor(vectorstore_service)
    router_chain = create_router_chain()
    
    ui.display_startup_message()
    
    chat_history = []

    while True:
        user_input = ui.get_user_input()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting.")
            break
        
        if user_input.lower() == "reindex":
            ui.display_system_message("🔄 Re-indexing workspace...")
            vectorstore_service.reindex()
            ui.display_system_message("✅ Workspace re-indexed successfully.", style="green")
            continue

        try:
            # --- Routing Logic ---
            ui.display_system_message("🤔 Analyzing request and routing to the best system...")
            route = router_chain.invoke({"input": user_input})

            if "ai_team" in route.lower():
                ui.display_system_message(f"🚀 Request is complex. Deploying the AI Team...")
                initial_state = {"task": user_input}
                for step in team_app.stream(initial_state):
                    step_name, step_output = list(step.items())[0]
                    ui.display_langgraph_step(step_name, step_output)
                chat_history.extend([("human", user_input), ("ai", "The AI team completed the task.")])

            else:
                ui.display_system_message(f"⚙️ Request is simple. Using the single agent...")
                last_tool_name = ""
                final_answer = ""
                ui.stream_final_answer("Agent")
                
                for event in single_agent_executor.stream({
                    "input": user_input,
                    "chat_history": chat_history
                }):
                    # --- THE FIX IS HERE ---
                    # Use .get() to safely access keys that might not exist
                    kind = event.get("event")
                    
                    if kind == "on_tool_start":
                        ui.console.print() # Add a newline to separate from the streaming answer
                        last_tool_name = event["name"]
                        ui.display_tool_start(event["name"], event["data"].get("input"))
                    
                    elif kind == "on_tool_end":
                        ui.display_tool_end(str(event["data"].get("output")), last_tool_name)
                        ui.stream_final_answer("Agent") # Prepare for the next streaming part
                    
                    elif kind == "on_chat_model_stream":
                        chunk = event["data"]["chunk"].content
                        if chunk:
                            final_answer += chunk
                            ui.stream_token(chunk)

                    elif kind == "on_agent_finish":
                        chat_history.extend([("human", user_input), ("ai", final_answer)])
                        ui.console.print() # Print a final newline
                
                # In case the agent finishes without a final stream (e.g., error)
                if not final_answer and "output" in event:
                     ui.stream_token(event["output"])
                     chat_history.extend([("human", user_input), ("ai", event["output"])])
                     ui.console.print()

        except Exception as e:
            ui.display_error(str(e))

if __name__ == "__main__":
    main()