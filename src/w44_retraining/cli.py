"""`w44` entry point: one subcommand per block, lazy imports."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="w44")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("harness", help="block 2: retraining-policy bench")
    sub.add_parser("churn", help="block 4: stability in closed form, validated by Monte Carlo")
    sub.add_parser("freshness", help="block 5: cadence per served quantile, on the calibrated DGP")
    sub.add_parser("gate", help="block 3: check configs/forecast_service.yaml (exit 1 if blocked)")
    sub.add_parser("serving", help="block 6: TSFM size by serving cost, licence gate")
    sub.add_parser("causal", help="block 8: retraining as an intervention, placebo, A/B, RDD")
    real = sub.add_parser("real", help="block 5 on real data (Monash Electricity Hourly)")
    real.add_argument("action", choices=["download", "run"])
    args = ap.parse_args(argv)

    if args.cmd == "harness":
        from .harness_study import print_study, study

        print_study(study())
    elif args.cmd == "freshness":
        from .freshness_study import print_study, study

        print_study(study())
    elif args.cmd == "real":
        from . import realdata

        if args.action == "download":
            print(realdata.download())
        else:
            realdata.print_study(realdata.study())
    elif args.cmd == "gate":
        from .gate import run

        raise SystemExit(0 if run() else 1)
    elif args.cmd == "serving":
        from .serving import print_study, study

        print_study(study())
    elif args.cmd == "causal":
        from .causal import print_study, study

        print_study(study())
    elif args.cmd == "churn":
        from .fundamentals import print_study, study

        print_study(study())


if __name__ == "__main__":
    main()
