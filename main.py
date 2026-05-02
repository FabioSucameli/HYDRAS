# Main Entry point 

from __future__ import annotations

from pollutant_gp.cli import parse_args
from pollutant_gp.workflow import run_workflow


def main() -> None:
    args = parse_args()
    run_workflow(args)


if __name__ == "__main__":
    main()

