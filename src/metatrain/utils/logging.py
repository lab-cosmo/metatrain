"""Logging."""

import contextlib
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from metatensor.torch.atomistic import ModelCapabilities

from .data import DatasetInfo
from .distributed.logging import is_main_process
from .external_naming import to_external_name
from .io import check_suffix
from .units import ev_to_mev, get_gradient_units


class MetricLogger:
    def __init__(
        self,
        log_obj: logging.Logger,
        dataset_info: Union[ModelCapabilities, DatasetInfo],
        initial_metrics: Union[Dict[str, float], List[Dict[str, float]]],
        names: Union[str, List[str]] = "",
    ):
        """
        Simple interface to log training metrics logging instance.

        Initialize the metric logger. The logger is initialized with the initial metrics
        and names relative to the metrics (e.g., "train", "validation").

        In this way, and by assuming that these metrics never increase, the logger can
        align the output to make it easier to read.

        :param log_obj: A logging instance
        :param model_outputs: outputs of the model. Used to infer physical quantities
            and units
        :param initial_metrics: initial training metrics
        :param names: names of the metrics (e.g., "train", "validation")
        """
        self.log_obj = log_obj

        # Length units will be used to infer units of forces/virials
        assert isinstance(dataset_info.length_unit, str)
        self.length_unit = dataset_info.length_unit

        # Save the model outputs. This will be useful to know
        # what physical quantities we are printing, along with their units
        if isinstance(dataset_info, DatasetInfo):
            self.model_outputs = dataset_info.targets
        elif isinstance(dataset_info, torch.ScriptObject):  # ModelCapabilities
            self.model_outputs = dataset_info.outputs
        else:
            raise ValueError(
                f"dataset_info must be of type `DatasetInfo` or `ModelCapabilities`, "
                f"not {type(dataset_info)}"
            )

        if isinstance(initial_metrics, dict):
            initial_metrics = [initial_metrics]
        if isinstance(names, str):
            names = [names]

        self.names = names

        # Since the quantities are supposed to decrease, we want to store the
        # number of digits at the start of the training, so that we can align
        # the output later:
        self.digits = {}
        for name, metrics_dict in zip(names, initial_metrics):
            for key, value in metrics_dict.items():
                target_name = key.split(" ", 1)[0]
                if "loss" in key:
                    # losses will be printed in scientific notation
                    continue
                unit = self._get_units(target_name)
                value, unit = ev_to_mev(value, unit)
                self.digits[f"{name}_{key}"] = _get_digits(value)

    def log(
        self,
        metrics: Union[Dict[str, float], List[Dict[str, float]]],
        epoch: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        """
        Log the metrics.

        The metrics are automatically aligned to make them easier to read, based on
        the order of magnitude of each metric given to the class at initialization.

        :param metrics: The current metrics to be printed.
        :param epoch: The current epoch (optional). If :py:class:`None`, the epoch
            will not be printed, and the logging string will start with the first
            metric in the ``metrics`` dictionary.
        :param rank: The rank of the process, if the training is distributed. In that
            case, the logger will only print the metrics for the process with rank 0.
        """

        if isinstance(metrics, dict):
            metrics = [metrics]

        if epoch is None:
            logging_string = ""
        else:
            # The epoch is printed with 4 digits, assuming that the training
            # will not last more than 9999 epochs
            logging_string = f"Epoch {epoch:4}"

        for name, metrics_dict in zip(self.names, metrics):
            for key, value in metrics_dict.items():

                new_key = key
                if key != "loss":  # special case: not a metric associated with a target
                    target_name, metric = new_key.split(" ", 1)
                    external_name = to_external_name(target_name, self.model_outputs)  # type: ignore # noqa: E501
                    new_key = f"{external_name} {metric}"

                if name == "":
                    logging_string += f", {new_key}: "
                else:
                    logging_string += f", {name} {new_key}: "
                if "loss" in key:  # print losses with scientific notation
                    logging_string += f"{value:.3e}"
                else:
                    unit = self._get_units(target_name)
                    value, unit = ev_to_mev(value, unit)
                    logging_string += (
                        f"{value:{self.digits[f'{name}_{key}'][0]}.{self.digits[f'{name}_{key}'][1]}f}"  # noqa: E501
                        + (f" {unit}" if unit != "" else "")
                    )

        # If there is no epoch, the string will start with a comma. Remove it:
        if logging_string.startswith(", "):
            logging_string = logging_string[2:]

        if rank is None or rank == 0:
            self.log_obj.info(logging_string)

    def _get_units(self, output: str) -> str:
        # Gets the units of an output
        if output.endswith("_gradients"):
            # handling <base_name>_<gradient_name>_gradients
            base_name = output[:-10]
            gradient_name = base_name.split("_")[-1]
            base_name = base_name.replace(f"_{gradient_name}", "")
            base_unit = self.model_outputs[base_name].unit
            unit = self._get_gradient_units(base_unit, gradient_name)
        else:
            unit = self.model_outputs[output].unit
        return unit

    def _get_gradient_units(self, base_unit: str, gradient_name: str) -> str:
        # Get the gradient units based on the unit of the base quantity
        # for example, if the base unit is "<unit>" and the gradient name is
        # "positions", the gradient unit will be "<unit>/<length_unit>".
        return get_gradient_units(base_unit, gradient_name, self.length_unit)


def _get_digits(value: float) -> Tuple[int, int]:
    """
    Finds the number of digits to print before and after the decimal point,
    based on the order of magnitude of the value.

    5 "significant" digits are guaranteed to be printed.

    :param value: The value for which the number of digits is calculated.
    """

    # Get order of magnitude of the value:
    order = int(np.floor(np.log10(value)))

    # Get the number of digits before the decimal point:
    if order < 0:
        digits_before = 1
    else:
        digits_before = order + 1

    # Get the number of digits after the decimal point:
    if order < 0:
        digits_after = 4 - order
    else:
        digits_after = max(1, 4 - order)

    total_characters = digits_before + digits_after + 1  # +1 for the point

    return total_characters, digits_after


@contextlib.contextmanager
def setup_logging(
    log_obj: logging.Logger,
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.WARNING,
):
    """Create a logging environment for a given ``log_obj``.

    Extracted and adjusted from
    github.com/MDAnalysis/mdacli/blob/main/src/mdacli/logger.py

    :param log_obj: A logging instance
    :param log_file: Name of the log file
    :param level: Set the root logger level to the specified level. If for example set
        to :py:obj:`logging.DEBUG` detailed debug logs inludcing filename and function
        name are displayed. For :py:obj:`logging.INFO` only the message logged from
        `errors`, `warnings` and `infos` will be displayed.
    """
    try:
        format = "[{asctime}][{levelname}]"
        if level == logging.DEBUG:
            format += ":{filename}:{funcName}:{lineno}"
        format += " - {message}"

        formatter = logging.Formatter(format, datefmt="%Y-%m-%d %H:%M:%S", style="{")
        handlers: List[Union[logging.StreamHandler, logging.FileHandler]] = []

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        handlers.append(stream_handler)

        if log_file:
            log_file = check_suffix(filename=log_file, suffix=".log")
            file_handler = logging.FileHandler(filename=str(log_file), encoding="utf-8")
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)

        # hide logging up to ERROR from secondary processes in distributed environments:
        if not is_main_process():
            level = logging.ERROR

        logging.basicConfig(format=format, handlers=handlers, level=level, style="{")
        logging.captureWarnings(True)

        if log_file:
            abs_path = str(Path(log_file).absolute().resolve())
            log_obj.info(f"This log is also available at {abs_path!r}.")
        else:
            log_obj.debug("Logging to file is disabled.")

        for handler in handlers:
            log_obj.addHandler(handler)

        yield

    finally:
        for handler in handlers:
            handler.flush()
            handler.close()
            log_obj.removeHandler(handler)


def get_cli_input(argv: Optional[List[str]] = None) -> str:
    """Propper formatted string of the command line input.

    :param argv: List of strings to parse. If :py:obj:`None` taken from
        :py:obj:`sys.argv`.
    """
    if argv is None:
        argv = sys.argv

    program_name = Path(argv[0]).name
    # Add additional quotes for connected arguments.
    arguments = [f'"{arg}"' if " " in arg else arg for arg in argv[1:]]
    return f"{program_name} {' '.join(arguments)}"
