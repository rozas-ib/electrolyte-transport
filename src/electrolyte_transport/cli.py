"""Command-line interface for electrolyte-transport."""

from .analysis import main as analysis_main


def main() -> None:
    analysis_main()
