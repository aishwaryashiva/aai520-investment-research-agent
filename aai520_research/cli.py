"""Command-line entry point for one research run."""

from __future__ import annotations

import argparse
from pathlib import Path

from aai520_research.agent import InvestmentResearchAgent
from aai520_research.memory import MemoryStore
from aai520_research.reporting import format_steps


def main(argv: list[str] | None = None) -> int:
    """Research one symbol and print the final brief."""
    parser = argparse.ArgumentParser(description="Run the AAI-520 investment research agent.")
    parser.add_argument("symbol", help="Ticker symbol, for example MSFT")
    parser.add_argument(
        "--memory",
        default=str(Path("data") / "memory" / "research_memory.json"),
        help="Path to the JSON memory file",
    )
    parser.add_argument("--revisions", type=int, default=2, help="Maximum optimizer revisions")
    args = parser.parse_args(argv)
    agent = InvestmentResearchAgent(
        memory=MemoryStore(args.memory),
        max_revisions=args.revisions,
    )
    result = agent.run(args.symbol)
    print(format_steps(result.plan))
    if result.gap_steps:
        print("\nFollow-up tools from self-reflection:")
        print(format_steps(result.gap_steps))
    print("\n" + result.final_markdown)
    print("\nReflection:")
    print(result.reflection.get("narrative"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
