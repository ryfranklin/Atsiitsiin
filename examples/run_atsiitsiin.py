from atsiitsiin.agents import AtsiiitsiinAgent
from atsiitsiin.config import AtsiiitsiinConfig


def show_help() -> None:
    """Display help information."""
    print("\n" + "=" * 60)
    print("Atsiitsʼiin - Help")
    print("=" * 60)
    print("\nCommands:")
    print("  help, /help        Show this help message")
    print("  exit, quit, /exit, /quit  Exit the program")
    print("\nUsage:")
    print("  You can interact with Atsiitsʼiin using natural language.")
    print(
        "  The agent will automatically decide when to store or retrieve memories."
    )
    print("\nExamples:")
    print("  > I want to remember meeting notes about Snowflake vector search")
    print("  > What did I say about vector search?")
    print("  > Tell me about my notes on Snowflake")
    print(
        "  > Remember: Python decorators are functions that modify other functions"
    )
    print("\n" + "=" * 60 + "\n")


def is_exit_command(text: str) -> bool:
    """Check if the input is an exit command."""
    exit_commands = ["exit", "quit", "/exit", "/quit"]
    return text.lower().strip() in exit_commands


def is_help_command(text: str) -> bool:
    """Check if the input is a help command."""
    help_commands = ["help", "/help"]
    return text.lower().strip() in help_commands


def main() -> None:
    """Run the interactive Atsiitsʼiin agent."""
    cfg = AtsiiitsiinConfig()
    agent = AtsiiitsiinAgent(cfg)

    print("=" * 60)
    print("Atsiitsʼiin - Natural Language Second Brain Agent")
    print("=" * 60)
    print("\nType 'help' for commands, or 'exit' to quit.")
    print("Interact naturally - the agent will store and retrieve memories.\n")

    while True:
        try:
            user_input = input("> ").strip()

            # Handle empty input
            if not user_input:
                continue

            # Handle exit commands
            if is_exit_command(user_input):
                print("\n👋 Goodbye!\n")
                break

            # Handle help commands
            if is_help_command(user_input):
                show_help()
                continue

            # Process through agent
            print("\n🤔 Processing...\n")
            response = agent.run(user_input)
            print(f"Agent: {response}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except EOFError:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
