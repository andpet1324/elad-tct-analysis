"""
Run one analysis by name. They all share the cleaned data, so after the first one
the others only run process() and plot().

    python3 tct.py x-scan data/*.csv -ped pedestals.csv -plots plots/run1 \
        -conf config.toml -cache caches/run1.parquet
    python3 tct.py knife-edge -ped pedestals.csv -plots plots/run1 \
        -conf config.toml -cache caches/run1.parquet
    python3 tct.py bias-scan ... --plot-only

All other arguments are the ones of BaseAnalysis, see `python3 tct.py x-scan -h`.
To add an analysis, put it in ANALYSES.
"""

import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).parent

# name on the command line: (file, class)
ANALYSES = {
    "x-scan": ("x-scan.py", "XScan"),
    "focus": ("focus.py", "Focus"),
    "knife-edge": ("knife-edge.py", "KnifeEdge"),
    "bias-scan": ("bias_scan.py", "BiasScan"),
}


def load(name: str):
    """The analysis class. The files have hyphens, so they are imported by path."""
    file, cls = ANALYSES[name]
    spec = importlib.util.spec_from_file_location(file.removesuffix(".py").replace("-", "_"), HERE / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, cls)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ANALYSES:
        names = ", ".join(ANALYSES)
        sys.exit(f"Usage: python3 tct.py <analysis> [arguments]\nAnalyses: {names}\n"
                 f"Arguments: python3 tct.py <analysis> -h")
    load(sys.argv[1])(argv=sys.argv[2:]).run()


if __name__ == "__main__":
    main()
