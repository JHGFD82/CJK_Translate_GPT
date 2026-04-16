"""Console output formatting utilities shared across the application."""


def print_section(title: str, content: str) -> None:
    """Print *content* wrapped in a ``=== Title ===`` header/footer block.

    Example output::

        === Translation ===
        ...content...
        ==================
    """
    bar = f"=== {title} ==="
    print(f"\n{bar}")
    print(content)
    print("=" * len(bar) + "\n")


def print_banner(title: str, width: int = 60) -> None:
    """Print a full-width ``=`` separator banner with *title* centred between two lines.

    Example output (width=60)::

        ============================================================
        TOKEN USAGE REPORT - PROFESSOR HELLER
        ============================================================
    """
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_subsection(label: str) -> None:
    """Print a subsection heading followed by a ``-`` rule.

    Example output::

        Model Breakdown:
        ----------------------------------------
    """
    print(f"\n{label}:")
    print("-" * 40)


def print_pass_result(label: str, content: str) -> None:
    """Print an intermediate pass result with a ``--- label ---`` header.

    Example output::

        --- Pass 1/3 result ---
        ...content...

    """
    print(f"\n--- {label} ---")
    print(content)
    print()
