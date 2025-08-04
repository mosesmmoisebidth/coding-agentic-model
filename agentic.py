from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import ShellTool
from langchain_experimental.tools import PythonREPLTool
from langchain_community.agent_toolkits.file_management.toolkit import FileManagementToolkit
from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_community.agent_toolkits import GitToolkit
# from langchain.tools import HumanInputRun
from tools.devops_tools import create_git_tool, create_docker_tool
from langchain_community.tools import HumanInputRun
from tools.codebase_qa_tool import CodebaseQATool
from services.vectorstore_service import VectorStoreService
import config

def create_agent_executor(vectorstore_service: VectorStoreService):
    """
    Creates and returns the agent executor.
    """
    # 1. Initialize LLM
    llm = ChatOpenAI(model=config.AGENT_MODEL, temperature=0)

    # git_toolkit = GitToolkit(repo_path=config.WORKING_DIR)
    # git_tools = git_toolkit.get_tools()
    # 2. Setup Tools
    file_tools = FileManagementToolkit(root_dir=config.WORKING_DIR).get_tools()
    shell_tool = ShellTool(working_directory=config.WORKING_DIR)
    python_tool = PythonREPLTool()

    git_tool = create_git_tool()
    docker_tool = create_docker_tool()
    
    # Our new custom RAG tool
    codebase_qa_tool = CodebaseQATool(vectorstore_service=vectorstore_service)

    web_search_tool = TavilySearchResults(
        name="web_search",
        description="A search engine. Use this to find real-time information, such as for new libraries, APIs, or error messages."
    )

    human_input_tool = HumanInputRun(
        name="ask_human_for_clarification",
        description="Use this to ask the human user a clarifying question. Use it when the user's request is ambiguous, you are unsure how to proceed, or you need more information to complete the task. The input to this tool should be the exact question you want to ask the user."
    )

    docker_tool = ShellTool(
        name="docker_tool",
        description=(
            "A tool for executing Docker commands. Use this for containerization tasks like:\n"
            "- `docker build -t <tag> .` to build a Docker image.\n"
            "- `docker run <image>` to run a container.\n"
            "- `docker push <tag>` to push an image.\n"
            "Ensure Docker Desktop or Docker Engine is running."
        ),
        working_directory=config.WORKING_DIR
    )

    tools = file_tools + [shell_tool, python_tool, codebase_qa_tool, web_search_tool, human_input_tool, docker_tool, git_tool]
    # 3. Create the Prompt
    # We are enhancing the system prompt to make the agent aware of its new RAG tool.
    system_prompt = """
        **YOUR PERSONA**
        You are Dev-GPT, a senior AI developer assistant. You are an expert programmer and a methodical problem-solver.
        You are operating in a sandboxed environment at the path: `{working_dir}`.

        **YOUR CORE DIRECTIVE**
        Your primary goal is to help the user by writing, testing, and debugging code. You must be proactive in verifying your own work and correcting your mistakes.

        **YOUR METHODOLOGY: THE PLAN-EXECUTE-TEST-REFLECT CYCLE**

        1.  **PLAN**: Before taking any action, think step-by-step. Analyze the user's request and create a clear, concise plan to achieve it.
        2.  **EXECUTE**: Carry out the steps in your plan using your available tools. Announce which step you are on (e.g., "Step 2: Writing the main function to `app.py`").
        3.  **TEST (CRITICAL STEP)**: After writing or modifying code, you **MUST** attempt to run or test it.
            - For Python code, use the `python_repl` tool to execute it.
            - For shell commands or running scripts, use the `shell` tool.
            - This step is not optional. You must verify your work.
        4.  **REFLECT & SELF-CORRECT**:
            - If the `TEST` step succeeds, great! Report the success to the user.
            - If the `TEST` step fails, **do not give up**. Analyze the error message returned by the tool.
            - Based on the error, update your `PLAN` to fix the issue.
            - Return to the `EXECUTE` step with the new, corrected plan.
            - If you are stuck on an error after a few attempts, use `web_search` to find a solution. If you are still stuck, use `ask_human_for_clarification`.

        **TOOL USAGE RULES**
        - `codebase_qa_tool`: Use this first for any questions about existing code.
        - `ask_human_for_clarification`: Use this for ambiguous requests, never for error debugging unless you have already tried to fix it yourself several times.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt.format(working_dir=config.WORKING_DIR)),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # 4. Create the Agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor