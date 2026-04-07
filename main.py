from __future__ import annotations

import sys
from pathlib import Path

try:
    from simple_term_menu import TerminalMenu
except Exception:
    TerminalMenu = None

from src.common.analysis import Analysis
from src.common.indexer import Indexer
from src.common.util import package_data, setup_data
from src.common.util.strings import snake_to_title


def _select_option(options: list[str], title: str) -> int | None:
    """Return the selected option index using a native menu or text prompt."""
    if TerminalMenu is not None and sys.stdin.isatty() and sys.stdout.isatty():
        menu = TerminalMenu(
            options,
            title=title,
            cycle_cursor=True,
            clear_screen=False,
        )
        return menu.show()

    print(title)
    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")

    if not sys.stdin.isatty():
        print("Interactive menu unavailable in this terminal. Rerun with a command argument.")
        return

    while True:
        try:
            raw_choice = input(f"Enter choice [1-{len(options)}] or press Enter to exit: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return None

        if not raw_choice:
            return None

        if raw_choice.isdigit():
            choice = int(raw_choice) - 1
            if 0 <= choice < len(options):
                return choice

        print("Invalid choice.")


def _print_named_options(label: str, values: list[str]) -> None:
    print(f"{label}:")
    for value in values:
        print(f"  - {value}")


def _run_analysis_instance(analysis: Analysis, output_dir: Path) -> None:
    print(f"\nRunning: {analysis.name}\n")
    saved = analysis.save(output_dir, formats=["png", "pdf", "csv", "json", "gif"])
    print("Saved files:")
    for fmt, path in saved.items():
        print(f"  {fmt}: {path}")


def analyze(name: str | None = None) -> None:
    """Run analysis by name or show interactive menu."""
    analyses = Analysis.load()

    if not analyses:
        print("No analyses found in src/analysis/")
        return

    output_dir = Path("output")

    # If name provided, run that specific analysis
    if name:
        if name == "all":
            print("\nRunning all analyses...\n")
            for analysis_cls in analyses:
                instance = analysis_cls()
                print(f"Running: {instance.name}")
                saved = instance.save(output_dir, formats=["png", "pdf", "csv", "json", "gif"])
                for fmt, path in saved.items():
                    print(f"  {fmt}: {path}")
            print("\nAll analyses complete.")
            return

        for analysis_cls in analyses:
            instance = analysis_cls()
            if instance.name == name:
                _run_analysis_instance(instance, output_dir)
                return

        print(f"Analysis '{name}' not found. Available analyses:")
        for analysis_cls in analyses:
            instance = analysis_cls()
            print(f"  - {instance.name}")
        sys.exit(1)

    options = ["[All] Run all analyses"]
    for analysis_cls in analyses:
        instance = analysis_cls()
        options.append(f"{snake_to_title(instance.name)}: {instance.description}")
    options.append("[Exit]")

    choice = _select_option(options, "Select an analysis to run:")

    if choice is None or choice == len(options) - 1:
        print("Exiting.")
        return

    if choice == 0:
        # Run all analyses
        print("\nRunning all analyses...\n")
        for analysis_cls in analyses:
            instance = analysis_cls()
            print(f"Running: {instance.name}")
            saved = instance.save(output_dir, formats=["png", "pdf", "csv", "json", "gif"])
            for fmt, path in saved.items():
                print(f"  {fmt}: {path}")
        print("\nAll analyses complete.")
    else:
        analysis_cls = analyses[choice - 1]
        instance = analysis_cls()
        _run_analysis_instance(instance, output_dir)


def index(name: str | None = None) -> None:
    """Run an indexer by name or show an interactive menu."""
    indexers = Indexer.load()

    if not indexers:
        print("No indexers found in src/indexers/")
        return

    if name:
        for indexer_cls in indexers:
            instance = indexer_cls()
            if instance.name == name:
                print(f"\nRunning: {instance.name}\n")
                instance.run()
                print("\nIndexer complete.")
                return

        print(f"Indexer '{name}' not found.")
        _print_named_options("Available indexers", [indexer_cls().name for indexer_cls in indexers])
        sys.exit(1)

    options = []
    for indexer_cls in indexers:
        instance = indexer_cls()
        options.append(f"{snake_to_title(instance.name)}: {instance.description}")
    options.append("[Exit]")

    choice = _select_option(options, "Select an indexer to run:")

    if choice is None or choice == len(options) - 1:
        print("Exiting.")
        return

    indexer_cls = indexers[choice]
    instance = indexer_cls()
    print(f"\nRunning: {instance.name}\n")
    instance.run()
    print("\nIndexer complete.")


def package():
    """Package the data directory into a zstd-compressed tar archive."""
    success = package_data()
    sys.exit(0 if success else 1)


def setup():
    """Download and extract the bundled data archive."""
    success = setup_data()
    sys.exit(0 if success else 1)


def main():
    if len(sys.argv) < 2:
        print("\nUsage: uv run main.py <command> [name]")
        print("Commands: analyze, index, package, setup")
        sys.exit(0)

    command = sys.argv[1]

    if command == "analyze":
        name = sys.argv[2] if len(sys.argv) > 2 else None
        analyze(name)
        sys.exit(0)

    if command == "index":
        name = sys.argv[2] if len(sys.argv) > 2 else None
        index(name)
        sys.exit(0)

    if command == "package":
        package()
        sys.exit(0)

    if command == "setup":
        setup()
        sys.exit(0)

    print(f"Unknown command: {command}")
    print("Commands: analyze, index, package, setup")
    sys.exit(1)


if __name__ == "__main__":
    main()
