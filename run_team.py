# run_team.py
import os
import config
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.tools import ShellTool
from langchain_community.agent_toolkits.file_management.toolkit import FileManagementToolkit
from team.state import TeamState
from team.agents import create_team_supervisor
# --- 1. SETUP ---
# Initialize the LLM and tools
llm = ChatOpenAI(model=config.AGENT_MODEL)
file_tools = FileManagementToolkit(root_dir=config.WORKING_DIR).get_tools()
shell_tool = ShellTool(working_directory=config.WORKING_DIR)
all_tools = file_tools + [shell_tool]

# Create the agents
agents = create_team_supervisor(llm, all_tools, file_tools)

# --- 2. DEFINE AGENT NODES ---
# Each node in the graph represents an agent performing an action.
def run_agent_node(state: TeamState, agent_key: str):
    agent = agents[agent_key]
    result = agent.invoke({"messages": [("user", state['task'])]})
    return {"agent_log": [f"Agent {agent_key} completed. Output: {result['output']}"]}

def architect_node(state: TeamState):
    agent = agents["Architect"]
    result = agent.invoke({"messages": [("user", state['task'])]})
    return {"plan": result['output'], "agent_log": [f"Architect created a plan: {result['output']}"]}

def coder_node(state: TeamState):
    agent = agents["Coder"]
    task_with_plan = f"Here is the plan:\n\n{state['plan']}\n\nPlease write the code."
    result = agent.invoke({"messages": [("user", task_with_plan)]})
    # For simplicity, we'll assume the coder mentions the file path in the output.
    # A more robust solution would parse this.
    return {"code": result['output'], "agent_log": [f"Coder wrote the code: {result['output']}"]}

def tester_node(state: TeamState):
    agent = agents["Tester"]
    task_for_tester = f"Here is the code to test:\n\n{state['code']}\n\nPlease write a pytest file and run it."
    result = agent.invoke({"messages": [("user", task_for_tester)]})
    return {"test_results": result['output'], "agent_log": [f"Tester ran tests. Results: {result['output']}"]}

def reviewer_node(state: TeamState):
    agent = agents["Reviewer"]
    task_for_reviewer = f"Here is the code to review:\n\n{state['code']}\n\nAnd here are the test results:\n{state['test_results']}"
    result = agent.invoke({"messages": [("user", task_for_reviewer)]})
    return {"review_comments": result['output'], "agent_log": [f"Reviewer provided feedback: {result['output']}"]}

# --- 3. DEFINE GRAPH LOGIC ---
# This defines how the team collaborates and moves from one step to the next.
def decide_after_test(state: TeamState):
    if "error" in state['test_results'].lower() or "fail" in state['test_results'].lower():
        print("Tests failed. Returning to Coder.")
        return "Coder"  # Go back to the coder to fix the code
    else:
        print("Tests passed. Proceeding to Reviewer.")
        return "Reviewer" # Proceed to the reviewer

def decide_after_review(state: TeamState):
    if "lgtm" in state['review_comments'].lower():
        print("Review passed. Project complete.")
        return END # The project is finished
    else:
        print("Review requires changes. Returning to Coder.")
        return "Coder" # Go back to the coder with the review feedback

# --- 4. BUILD THE GRAPH ---
workflow = StateGraph(TeamState)

workflow.add_node("Architect", architect_node)
workflow.add_node("Coder", coder_node)
workflow.add_node("Tester", tester_node)
workflow.add_node("Reviewer", reviewer_node)

workflow.set_entry_point("Architect")

workflow.add_edge("Architect", "Coder")
workflow.add_edge("Coder", "Tester")

workflow.add_conditional_edges(
    "Tester",
    decide_after_test,
    {"Coder": "Coder", "Reviewer": "Reviewer"}
)
workflow.add_conditional_edges(
    "Reviewer",
    decide_after_review,
    {"Coder": "Coder", END: END}
)

# Compile the graph into a runnable application
app = workflow.compile()

# --- 5. RUN THE TEAM ---
if __name__ == "__main__":
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    print("🤖 AI Team is ready. Enter a complex task for them to complete.")
    
    user_task = input("\n🗣️  You: ")
    
    if user_task:
        initial_state = {"task": user_task}
        # The `stream` method lets us see the output from each step as it happens
        for step in app.stream(initial_state):
            step_name, step_output = list(step.items())[0]
            print(f"--- AGENT: {step_name} ---")
            print(f"Log: {step_output.get('agent_log', 'No log entry')[-1]}")
            print("\n")