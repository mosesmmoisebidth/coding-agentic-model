from langchain_community.tools import ShellTool
import config
def create_git_tool():
    """
    Creates a specialized tool for Git operations.
    This is a pre-configured ShellTool.
    """
    description = (
        "A tool for executing Git commands. Use this for version control tasks like:"
        "\n- `git init` to initialize a repository."
        "\n- `git add <file>` to stage changes."
        "\n- `git commit -m '<message>'` to commit changes."
        "\n- `git push` to push changes to a remote repository."
        "\n- `git pull` to pull changes."
        "\n- `git branch` to manage branches."
        "\nAlways run in the project's root directory."
    )
    git_tool = ShellTool(
        name="git_tool",
        description=description,
        working_directory=config.WORKING_DIR
    )
    return git_tool

def create_docker_tool():
    """
    Creates a specialized tool for Docker operations.
    This is a pre-configured ShellTool.
    """
    description = (
        "A tool for executing Docker commands. Use this for containerization tasks like:"
        "\n- `docker build -t <tag> .` to build a Docker image from a Dockerfile."
        "\n- `docker run <image>` to run a container."
        "\n- `docker push <tag>` to push an image to a container registry."
        "\n- `docker ps` to list running containers."
        "\nEnsure Docker Desktop or Docker Engine is running on the system."
    )
    docker_tool = ShellTool(
        name="docker_tool",
        description=description,
        working_directory=config.WORKING_DIR
    )
    return docker_tool