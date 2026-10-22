#!/usr/bin/env python3
"""Unified benchmark evaluation entry point."""
from pathlib import Path
import argparse
import importlib.util
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--benchmark", required=True, choices=("geneval2", "geneval", "genevalpp"))
    parser.add_argument("-h", "--help", action="store_true")
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv in (["--help"], ["-h"]):
        parser.print_help()
        return 0
    args, remaining = parser.parse_known_args(argv)
    if args.benchmark != "geneval2":
        parser.error(f"{args.benchmark}: evaluator adapter is not implemented")
    path = Path(__file__).resolve().parents[1] / "evaluation/geneval2/scorer.py"
    spec = importlib.util.spec_from_file_location("geneval2_scorer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main(remaining + (["--help"] if args.help else []))


if __name__ == "__main__":
    raise SystemExit(main())
