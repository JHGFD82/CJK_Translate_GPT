"""Console output formatting utilities shared across runtime modules."""


def print_section(title: str, content: str) -> None:
    """Print *content* wrapped in a consistent ``=== Title ===`` header/footer block."""
    bar = f"=== {title} ==="
    print(f"\n{bar}")
    print(content)
    print("=" * len(bar) + "\n")
