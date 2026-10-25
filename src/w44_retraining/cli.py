"""`w44` entry point: one subcommand per block, lazy imports."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="w44")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("harness", help="block 2: retraining-policy bench")
    args = ap.parse_args(argv)

    if args.cmd == "harness":
        from .harness_study import print_study, study

        print_study(study())


if __name__ == "__main__":
    main()
