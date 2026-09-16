"""Command-line entry point.

    cortica run <pipeline.yaml> <input> [--out data_raw.fif] [--report out.html]
    cortica                      # launch the desktop GUI (needs the [gui] extra)

The ``run`` command is fully headless — the same pipeline authored in the GUI can
be re-run on new data here, which is the reproducibility guarantee in practice.
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cortica",
        description="Cortica — a guided, reproducible workbench for EEG & fNIRS.",
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run a pipeline on an input recording (headless).")
    run_p.add_argument("pipeline", help="Path to a .pipeline.yaml file")
    run_p.add_argument("input", help="Path to the input recording (fif, edf, snirf, ...)")
    run_p.add_argument("--out", help="Write the processed recording here (e.g. out_raw.fif)")
    run_p.add_argument("--report", help="Write an HTML report here")

    args = parser.parse_args(argv)

    if args.command == "run":
        return _run(args)
    return _launch_gui()


def _run(args) -> int:
    import cortica.steps  # noqa: F401  register the built-in steps
    from cortica import io
    from cortica.core.pipeline import Pipeline
    from cortica.core.registry import default_registry

    with open(args.pipeline) as handle:
        pipeline = Pipeline.from_yaml(handle.read())

    dataset = io.load_raw(args.input)
    result = pipeline.run(dataset, default_registry)

    if args.out:
        result.payload.save(args.out, overwrite=True, verbose=False)
    if args.report:
        from cortica.report import build_report

        build_report(result, pipeline, args.report)

    print(f"Ran {len(pipeline.steps)} step(s) on {args.input}")
    if args.out:
        print(f"  wrote {args.out}")
    if args.report:
        print(f"  wrote {args.report}")
    return 0


def _launch_gui() -> int:
    try:
        from cortica.gui.app import run_app
    except ImportError:
        print(
            "The Cortica desktop GUI needs the 'gui' extra. Install it with:\n"
            "    pip install cortica[gui]",
            file=sys.stderr,
        )
        return 1

    return run_app()


if __name__ == "__main__":
    raise SystemExit(main())
