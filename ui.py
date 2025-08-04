# ui.py
import ast
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
import os

class UI:
    """
    Manages all user interface interactions, including input and output.
    """
    def __init__(self):
        self.console = Console()
        history_file = os.path.join(os.path.expanduser("~"), ".dev_agent_history")
        self.session = PromptSession(history=FileHistory(history_file))

    def display_startup_message(self):
        """Displays the initial welcome message."""
        # ... (This function remains the same)
        welcome_text = Markdown("# 🤖 Unified AI Agent is Ready!\n*   Full DevOps, Codebase Awareness, and Web Search.\n*   I will stream my thoughts and actions for full transparency.\n*   Type `exit` or `quit` to end.")
        panel = Panel(welcome_text, title="[bold green]Welcome[/bold green]", border_style="green")
        self.console.print(panel)

    def get_user_input(self) -> str:
        """Gets user input using prompt_toolkit for a rich experience with history."""
        return self.session.prompt("🗣️ You: ")

    def display_agent_response(self, response: str, agent_name: str):
        """Displays the agent's final response (non-streaming)."""
        # ... (This function remains the same)
        panel = Panel(Markdown(response, style="default"), title=f"[bold blue]🤖 {agent_name}[/bold blue]", border_style="blue")
        self.console.print(panel)

    def stream_final_answer(self, agent_name: str):
        """Prepares the console to stream the final answer."""
        self.console.print(f"[bold blue]🤖 {agent_name}:[/bold blue] ", end="")

    def stream_token(self, token: str):
        """Prints a single token to the console as part of a stream."""
        self.console.print(token, end="", style="default")

    def display_tool_start(self, tool_name: str, input_str: str):
        """Displays a notification that a tool is about to be called."""
        panel = Panel(
            Markdown(f"**Input:**\n```\n{input_str}\n```"),
            title=f"[bold yellow]🛠️ Using Tool: {tool_name}[/bold yellow]",
            border_style="yellow"
        )
        self.console.print(panel)

    def display_tool_end(self, output: str, tool_name: str):
        """Displays the result of a tool call, formatting it based on the tool."""
        if tool_name == "web_search":
            self._display_web_search_results(output)
        else:
            # For all other tools, display the output as syntax-highlighted code/text
            panel = Panel(
                Syntax(output, "python", theme="monokai", word_wrap=True),
                title=f"[bold green]✅ Tool Output: {tool_name}[/bold green]",
                border_style="green"
            )
            self.console.print(panel)

    def _display_web_search_results(self, output: str):
        """Formats and displays web search results in a readable table."""
        try:
            # The output from the tool is a string representation of a list of dicts.
            # We use ast.literal_eval for safe evaluation.
            results = ast.literal_eval(output)
        except (ValueError, SyntaxError):
            self.console.print("[red]Error parsing web search results.[/red]")
            return

        table = Table(title="🌐 Web Search Results", border_style="cyan", show_lines=True)
        table.add_column("Rank", style="magenta", justify="center")
        table.add_column("Title", style="cyan")
        table.add_column("Content Snippet", style="default")

        for i, result in enumerate(results[:3], 1): # Show top 3 results
            title = f"[link={result.get('url', '')}]{result.get('title', 'No Title')}[/link]"
            content = result.get('content', '')
            # Truncate content for readability
            snippet = content[:250] + "..." if len(content) > 250 else content
            table.add_row(str(i), Markdown(title), snippet)
        
        self.console.print(table)


    def display_system_message(self, message: str, style="yellow"):
        """Displays a system message."""
        self.console.print(f"[{style}]⚙️ {message}[/{style}]")

    def display_error(self, error_message: str):
        """Displays an error message."""
        panel = Panel(f"[default]{error_message}[/default]", title="[bold red]❌ Error[/bold red]", border_style="red")
        self.console.print(panel)

    def display_langgraph_step(self, step_name: str, output: dict):
        """Displays the output of a LangGraph step."""
        # ... (This function can remain the same)
        log_entry = "No log entry"
        if 'agent_log' in output and output['agent_log']:
            log_entry = output['agent_log'][-1]
        panel = Panel(Syntax(str(log_entry), "python", theme="monokai", line_numbers=False), title=f"[bold magenta]🚀 Team Step: {step_name}[/bold magenta]", border_style="magenta")
        self.console.print(panel)