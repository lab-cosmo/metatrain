"""The main entry point for the metatensor-models command line interface."""

import argparse
import logging
import os
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path

from . import __version__
from .cli.eval import _add_eval_model_parser, eval_model
from .cli.export import _add_export_model_parser
from .cli.train import _add_train_model_parser, train_model
from .utils.io import export
from .utils.logging import setup_logging


logger = logging.getLogger(__name__)


def _datetime_output_path(now: datetime) -> Path:
    """Get a date and time based output path."""
    return Path(
        "outputs",
        now.strftime("%Y-%m-%d"),
        now.strftime("%H-%M-%S"),
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    if len(sys.argv) < 2:
        ap.error("You must specify a sub-command")

    # If you change the synopsis of these commands or add new ones adjust the completion
    # script at `src/metatensor/models/share/metatensor-models-completion.bash`.
    ap.add_argument(
        "--version",
        action="version",
        version=f"metatensor-models {__version__}",
    )

    ap.add_argument(
        "--debug",
        action="store_true",
        help="Run with debug options.",
    )

    ap.add_argument(
        "--shell-completion",
        action="version",
        help="Path to the shell completion script",
        version=str(Path(__file__).parent / "share/metatensor-models-completion.bash"),
    )

    # Add sub-parsers
    subparser = ap.add_subparsers(help="sub-command help")
    _add_eval_model_parser(subparser)
    _add_export_model_parser(subparser)
    _add_train_model_parser(subparser)

    args = ap.parse_args()
    callable = args.__dict__.pop("callable")
    debug = args.__dict__.pop("debug")

    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
        warnings.filterwarnings("ignore")  # ignore all warnings if not in debug mode

    # Only save log to file for train command
    if callable == "train_model":
        checkpoint_dir = _datetime_output_path(now=datetime.now())
        os.makedirs(checkpoint_dir, exist_ok=True)  # exist_ok=True for distributed
        logfile = checkpoint_dir / "train.log"

        args.__dict__["checkpoint_dir"] = checkpoint_dir
    else:
        logfile = None

    with setup_logging(logger, logfile=logfile, level=level):
        try:
            if callable == "eval_model":
                eval_model(**args.__dict__)
            elif callable == "export_model":
                export(**args.__dict__)
            elif callable == "train_model":
                train_model(**args.__dict__)
            else:
                raise ValueError("internal error when selecting a sub-command.")
        except Exception as e:
            if debug:
                traceback.print_exc()
            else:
                sys.exit(f"\033[31mERROR: {e}\033[0m")  # format error in red!


if __name__ == "__main__":
    main()
