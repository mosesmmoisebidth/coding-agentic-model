# main.py
import os
import config
from agentic import create_agent_executor
from ui import UI
from services.vectorstore_service import VectorStoreService
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from run_team import app as team_app

def create_router_chain():
    # ... (no changes needed in this function)
    router_prompt_template = """
        You are an expert at analyzing user requests for a software development AI agent.
        Your task is to classify the user's request into one of two categories:

        1.  **single_agent**: For simple, direct, and short tasks. These are usually single-file operations, quick questions, or one-off commands.
            Examples:
            - "What is the capital of France?"
            - "List the files in the current directory."
            - "Write 'hello world' to a file named 'hello.py'."
            - "What does the function `calculate_total` in `utils.py` do?"
            - "What is the latest version of Flask?"
            - "Write the cpp code to calculate the factorial of a number"
            - "Write the simple node.j code to create simple web server"
            - "Write the python code to create and reverse the linked list"
            - "Write the algorithm to find the shortest path in a graph"

        2.  **ai_team**: For complex, multi-step tasks that require planning, creating multiple files, testing, and reviewing. These are typically project-level requests.
            Examples:
            - "Build a simple blog application using Flask."
            - "Create a new FastAPI endpoint for user authentication, and then write a pytest file to test it."
            - "Refactor the entire 'database.py' module to use a connection pool and then verify the changes."
            - "Develop a web scraper to get data from a website and save it to a CSV file."
            - "Create a Dockerfile for the current project and write a test to ensure it builds correctly"
            - "Analyse this code and draw the graph showing how it flows"
            - "Analyse the following algorithms and give their time complexity and space complexity"
            
        Based on the user's request below, respond with ONLY the category name ('single_agent' or 'ai_team') and nothing else.

        User Request:
        "{input}"
    """
    llm = ChatOpenAI(model=config.AGENT_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_template(router_prompt_template)
    return prompt | llm | StrOutputParser()

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
    ui = UI()
    print(f"✅ Agent working directory set to: {config.WORKING_DIR}")
    # Initialize our RAG service
    vectorstore_service = VectorStoreService(
        working_dir=config.WORKING_DIR,
        supported_file_types=config.SUPPORTED_FILE_TYPES,
        embeddings_model=config.EMBEDDINGS_MODEL
    )
    single_agent_executor = create_agent_executor(vectorstore_service)
    router_chain = create_router_chain()
    ui.display_startup_message()
    # Create the agent executor
    chat_history = []

    while True:
        user_input = input("\n🗣️  You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting.")
            break
        
        # Add a special command to trigger re-indexing
        if user_input.lower() == "reindex":
            ui.display_system_message("🔄 Re-indexing workspace...")
            vectorstore_service.reindex()
            ui.display_system_message("✅ Workspace re-indexed successfully.", style="green")
            continue

        try:
            ui.display_system_message("🤔 Analyzing request and routing to the best system...")
            route = router_chain.invoke({"input": user_input})
            # response = router_chain.invoke({
            #     "input": user_input,
            #     "chat_history": chat_history
            # })

            if "ai_team" in route.lower():
                ui.display_system_message(f"🚀 Request is complex. Deploying the AI Team...")
                initial_state = {"task": user_input}
                final_state = None
                for step in team_app.stream(initial_state):
                    step_name, step_output = list(step.items())[0]
                    # trunk-ignore(ruff/F841)
                    final_state = step_output
                    ui.display_langgraph_step(step_name, step_output) # <-- NEW LANGGRAPH DISPLAY
                final_response = "The AI team has completed the task."
                if final_state.get('review_comments'):
                    final_response += f"- Review: {final_state['review_comments']}\n"
                if final_state.get('test_results'):
                    final_response += f"- Test Results: {final_state['test_results']}\n"
                if final_state.get('code'):
                    final_response += f"- Final Code Snippet: \n{final_state['code']}"
                chat_history.extend([("human", user_input), ("ai", final_response)])

            else:
                ui.display_system_message(f"⚙️ Request is simple. Using the single agent...")
                
                # --- NEW STREAMING LOGIC FOR MODERN AGENTS ---
                final_answer = ""
                
                # Use a with block for the status to ensure it's removed on completion/error
                with ui.console.status("[bold green]Agent is thinking...", spinner="dots") as status:
                    for chunk in single_agent_executor.stream({
                        "input": user_input,
                        "chat_history": chat_history
                    }):
                        # Check for actions (tool calls)
                        if "actions" in chunk:
                            status.stop() # Stop the spinner
                            for action in chunk["actions"]:
                                ui.display_tool_start(action.tool, str(action.tool_input))
                            status.start() # Restart the spinner while tool runs

                        # Check for steps (tool outputs)
                        elif "steps" in chunk:
                            status.stop() # Stop the spinner
                            for step in chunk["steps"]:
                                ui.display_tool_end(str(step.observation), step.action.tool)
                            status.start() # Restart the spinner

                        # Check for the final answer chunk
                        elif "output" in chunk:
                            status.stop() # Stop the spinner permanently
                            final_answer += chunk["output"]
                            # The final answer comes in one go, so we display it directly
                            # instead of streaming token by token.
                ui.display_agent_response(final_answer, "Agent")
                chat_history.extend([("human", user_input), ("ai", final_answer)])
        except Exception as e:
            # trunk-ignore(git-diff-check/error)
            ui.display_error(str(e)) 
            # print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()