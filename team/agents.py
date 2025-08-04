# team/agents.py
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

CORE_AGENT_CONSTITUTION = """
    **Core Directives:**
    1.  **Never Give Up**: You are persistent and resourceful. If you encounter an error, you will analyze it, and try to fix it.
    2.  **Self-Correction Loop**:
        a.  If a tool returns an error (e.g., from running code), first analyze the error message.
        b.  If it's a simple syntax error or a mistake you made, correct your plan and try again.
        c.  If the error is unfamiliar or complex, use the `web_search` tool to find information about the error.
        d.  After researching, update your plan and re-execute the corrected steps.
    3.  **Ask for Help when Truly Stuck**: If you have tried to self-correct multiple times and are still failing, or if you encounter a system-level problem you cannot solve (e.g., a missing compiler, permissions issues), use the `ask_human_for_clarification` tool. Clearly state the problem and what you have already tried to do.
    """

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str, agent_name: str) -> AgentExecutor:
    """Helper function to create an agent executor."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(name=agent_name, agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

def create_team_supervisor(llm: ChatOpenAI, all_tools: list, file_tools: list):
    """
    Creates the supervisor and all specialized agents for the team.
    """
    # Define system prompts for each agent
    architect_prompt = (
        "You are an expert software architect. Your role is to take a high-level development task "
        "and create a detailed, step-by-step technical plan. "
        "**Crucially, you must first decide on the best programming language and technologies for the task** and specify them in the plan. "
        "The plan must be clear, concise, and cover all necessary files, functions, and logic. "
        "You do not write code or test. Your only output is the plan."
    )
    
    coder_prompt = (
        "**Your Role**: You are an expert **polyglot programmer**, fluent in all programming languages. "
        "Your task is to take a technical plan and write the code in the **exact language and framework specified in the plan**. "
        "You must follow the plan precisely. Use the file management tools to write the code to the specified files. "
        "Ensure the code is clean, efficient, and well-commented.\n\n"
        + CORE_AGENT_CONSTITUTION
    )
    
    tester_prompt = (
        "**Your Role**: You are a meticulous Quality Assurance (QA) engineer. Your task is to test the code written by the Coder. "
        "1. Identify the programming language from the plan or the code files. "
        "2. Write a comprehensive test file using the **standard testing framework for that language** (e.g., `pytest` for Python, `Jest` for JavaScript/TypeScript, `JUnit` for Java, etc.). "
        "3. Use the `shell` tool to install any needed dependencies and run the tests. "
        "4. Report the test results. If there are failures, provide the full error output. "
        "Do not proceed to the Reviewer if tests are failing.\n\n"
        + CORE_AGENT_CONSTITUTION
    )

    reviewer_prompt = (
        "You are a senior developer and code reviewer. Your role is to review the code for quality, "
        "style, and best practices relevant to the language it is written in. "
        "If the code is perfect and ready for production, respond with ONLY the word 'LGTM' (Looks Good To Me). "
        "Otherwise, provide specific, actionable comments for improvement."
    )
    
    # Create the agents
    architect_agent = create_agent(llm, [], architect_prompt, "Architect")
    coder_agent = create_agent(llm, file_tools, coder_prompt, "Coder")
    tester_agent = create_agent(llm, all_tools, tester_prompt, "Tester")
    reviewer_agent = create_agent(llm, file_tools, reviewer_prompt, "Reviewer")

    return {
        "Architect": architect_agent,
        "Coder": coder_agent,
        "Tester": tester_agent,
        "Reviewer": reviewer_agent,
    }