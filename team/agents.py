# team/agents.py
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import config

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str) -> AgentExecutor:
    """Helper function to create an agent executor."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)

def create_team_supervisor(llm: ChatOpenAI, all_tools: list, file_tools: list):
    """
    Creates the supervisor and all specialized agents for the team.
    """
    # Define system prompts for each agent
    architect_prompt = (
        "You are an expert software architect. Your role is to take a high-level development task "
        "and create a detailed, step-by-step technical plan for the Coder agent. "
        "The plan should be clear, concise, and cover all necessary files, functions, and logic. "
        "Do not write any code."
    )
    
    coder_prompt = (
        "You are an expert Python programmer. Your role is to take a technical plan and write the code. "
        "You must follow the plan exactly. Use the file management tools to write the code to the specified files. "
        "Ensure the code is clean, efficient, and well-commented."
    )
    
    tester_prompt = (
        "You are a Quality Assurance (QA) engineer. Your role is to test the code written by the Coder. "
        "You must write a test file using the `pytest` framework, then execute the tests using the shell tool. "
        "The test file should be written to the workspace. "
        "Report the test results, including any errors or failures."
    )

    reviewer_prompt = (
        "You are a senior developer and code reviewer. Your role is to review the code for quality, "
        "style, and adherence to best practices. Provide constructive feedback. "
        "If the code is good, respond with 'LGTM' (Looks Good To Me). "
        "Otherwise, provide specific comments for improvement."
    )
    
    # Create the agents
    architect_agent = create_agent(llm, [], architect_prompt)
    coder_agent = create_agent(llm, file_tools, coder_prompt)
    tester_agent = create_agent(llm, all_tools, tester_prompt)
    reviewer_agent = create_agent(llm, file_tools, reviewer_prompt)

    return {
        "Architect": architect_agent,
        "Coder": coder_agent,
        "Tester": tester_agent,
        "Reviewer": reviewer_agent,
    }