import argparse
import itertools
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import metatensor.torch
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import MetatensorAtomisticModel
from omegaconf import DictConfig, OmegaConf

from ..utils.data import (
    Dataset,
    TargetInfo,
    TargetInfoDict,
    collate_fn,
    read_systems,
    read_targets,
    write_predictions,
)
from ..utils.errors import ArchitectureError
from ..utils.evaluate_model import evaluate_model
from ..utils.logging import MetricLogger
from ..utils.metrics import RMSEAccumulator
from ..utils.neighbor_lists import get_system_with_neighbor_lists
from ..utils.omegaconf import expand_dataset_config
from ..utils.per_atom import average_by_num_atoms
from .formatter import CustomHelpFormatter


logger = logging.getLogger(__name__)


def _add_eval_model_parser(subparser: argparse._SubParsersAction) -> None:
    """Add the `eval_model` paramaters to an argparse (sub)-parser"""

    if eval_model.__doc__ is not None:
        description = eval_model.__doc__.split(r":param")[0]
    else:
        description = None

    # If you change the synopsis of these commands or add new ones adjust the completion
    # script at `src/metatrain/share/metatrain-completion.bash`.
    parser = subparser.add_parser(
        "eval",
        description=description,
        formatter_class=CustomHelpFormatter,
    )
    parser.set_defaults(callable="eval_model")
    parser.add_argument(
        "path",
        type=str,
        help="Saved exported model to be evaluated.",
    )
    parser.add_argument(
        "options",
        type=str,
        help="Eval options file to define a dataset for evaluation.",
    )
    parser.add_argument(
        "-e",
        "--extensions-dir",
        type=str,
        required=False,
        dest="extensions_directory",
        default=None,
        help=(
            "path to a directory containing all extensions required by the exported "
            "model"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        required=False,
        default="output.xyz",
        help="filename of the predictions (default: %(default)s)",
    )


def _prepare_eval_model_args(args: argparse.Namespace) -> None:
    """Prepare arguments for eval_model."""
    args.options = OmegaConf.load(args.options)
    args.model = metatensor.torch.atomistic.load_atomistic_model(
        path=args.__dict__.pop("path"),
        extensions_directory=args.__dict__.pop("extensions_directory"),
    )


def _concatenate_tensormaps(
    tensormap_dict_list: List[Dict[str, TensorMap]]
) -> Dict[str, TensorMap]:
    # Concatenating TensorMaps is tricky, because the model does not know the
    # "number" of the system it is predicting. For example, if a model predicts
    # 3 batches of 4 atoms each, the system labels will be [0, 1, 2, 3],
    # [0, 1, 2, 3], [0, 1, 2, 3] for the three batches, respectively. Due
    # to this, the join operation would not achieve the desired result
    # ([0, 1, 2, ..., 11, 12]). Here, we fix this by renaming the system labels.

    system_counter = 0
    n_systems = 0
    tensormaps_shifted_systems = []
    for tensormap_dict in tensormap_dict_list:
        tensormap_dict_shifted = {}
        for name, tensormap in tensormap_dict.items():
            new_keys = []
            new_blocks = []
            for key, block in tensormap.items():
                new_key = key
                where_system = block.samples.names.index("system")
                n_systems = torch.max(block.samples.column("system")) + 1
                new_samples_values = block.samples.values
                new_samples_values[:, where_system] += system_counter
                new_block = TensorBlock(
                    values=block.values,
                    samples=Labels(block.samples.names, values=new_samples_values),
                    components=block.components,
                    properties=block.properties,
                )
                for gradient_name, gradient_block in block.gradients():
                    new_block.add_gradient(
                        gradient_name,
                        gradient_block,
                    )
                new_keys.append(new_key)
                new_blocks.append(new_block)
            tensormap_dict_shifted[name] = TensorMap(
                keys=Labels(
                    names=tensormap.keys.names,
                    values=torch.stack([new_key.values for new_key in new_keys]),
                ),
                blocks=new_blocks,
            )
        tensormaps_shifted_systems.append(tensormap_dict_shifted)
        system_counter += n_systems

    return {
        target: metatensor.torch.join(
            [pred[target] for pred in tensormaps_shifted_systems], axis="samples"
        )
        for target in tensormaps_shifted_systems[0].keys()
    }


def _eval_targets(
    model: Union[MetatensorAtomisticModel, torch.jit._script.RecursiveScriptModule],
    dataset: Union[Dataset, torch.utils.data.Subset],
    options: TargetInfoDict,
    return_predictions: bool,
) -> Optional[Dict[str, TensorMap]]:
    """Evaluates an exported model on a dataset and prints the RMSEs for each target.
    Optionally, it also returns the predictions of the model.

    Wraps around metatrain.cli.evaluate_model.
    """

    if len(dataset) == 0:
        logger.info("This dataset is empty. No evaluation will be performed.")

    # Attach neighbor lists to the systems:
    # TODO: these might already be present... find a way to avoid recomputing
    # if already present (e.g. if this function is called after training)
    for sample in dataset:
        system = sample["system"]
        get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

    # Infer the device from the model
    device = next(itertools.chain(model.parameters(), model.buffers())).device

    # Create a dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # TODO: allow to set from outside!!
        collate_fn=collate_fn,
        shuffle=False,
    )

    # Initialize RMSE accumulator:
    rmse_accumulator = RMSEAccumulator()

    # If we're returning the predictions, we need to store them:
    if return_predictions:
        all_predictions = []

    # Evaluate the model
    for batch in dataloader:
        systems, batch_targets = batch
        systems = [system.to(device=device) for system in systems]
        batch_targets = {
            key: value.to(device=device) for key, value in batch_targets.items()
        }
        batch_predictions = evaluate_model(model, systems, options, is_training=False)
        batch_predictions = average_by_num_atoms(
            batch_predictions, systems, per_structure_keys=[]
        )
        batch_targets = average_by_num_atoms(
            batch_targets, systems, per_structure_keys=[]
        )
        rmse_accumulator.update(batch_predictions, batch_targets)
        if return_predictions:
            all_predictions.append(batch_predictions)

    # Finalize the RMSEs
    rmse_values = rmse_accumulator.finalize(not_per_atom=["positions_gradients"])
    # print the RMSEs with MetricLogger
    metric_logger = MetricLogger(
        log_obj=logger,
        dataset_info=model.capabilities(),
        initial_metrics=rmse_values,
    )
    metric_logger.log(rmse_values)

    if return_predictions:
        # concatenate the TensorMaps
        all_predictions_joined = _concatenate_tensormaps(all_predictions)
        return all_predictions_joined
    else:
        return None


def eval_model(
    model: Union[MetatensorAtomisticModel, torch.jit._script.RecursiveScriptModule],
    options: DictConfig,
    output: Union[Path, str] = "output.xyz",
) -> None:
    """Evaluate an exported model on a given data set.

    If ``options`` contains a ``targets`` sub-section, RMSE values will be reported. If
    this sub-section is missing, only a xyz-file with containing the properties the
    model was trained against is written.

    :param model: Saved model to be evaluated.
    :param options: DictConfig to define a test dataset taken for the evaluation.
    :param output: Path to save the predicted values
    """
    logger.info("Setting up evaluation set.")

    # TODO: once https://github.com/lab-cosmo/metatensor/pull/551 is merged and released
    # use capabilities instead of this workaround
    dtype = next(model.parameters()).dtype

    if isinstance(output, str):
        output = Path(output)

    options_list = expand_dataset_config(options)
    for i, options in enumerate(options_list):
        if len(options_list) == 1:
            extra_log_message = ""
            file_index_suffix = ""
        else:
            extra_log_message = f" with index {i}"
            file_index_suffix = f"_{i}"
        logger.info(f"Evaluating dataset{extra_log_message}")

        eval_systems = read_systems(
            filename=options["systems"]["read_from"],
            fileformat=options["systems"]["file_format"],
            dtype=dtype,
        )

        if hasattr(options, "targets"):
            # in this case, we only evaluate the targets specified in the options
            # and we calculate RMSEs
            eval_targets, eval_info_dict = read_targets(options["targets"], dtype=dtype)
        else:
            # in this case, we have no targets: we evaluate everything
            # (but we don't/can't calculate RMSEs)
            # TODO: allow the user to specify which outputs to evaluate
            eval_targets = {}
            eval_info_dict = TargetInfoDict()
            gradients = {"positions"}
            if all(not torch.all(system.cell == 0) for system in eval_systems):
                # only add strain if all structures have cells
                gradients.add("strain")
            for key in model.capabilities().outputs.keys():
                eval_info_dict[key] = TargetInfo(
                    quantity=model.capabilities().outputs[key].quantity,
                    unit=model.capabilities().outputs[key].unit,
                    per_atom=False,  # TODO: allow the user to specify this
                    gradients=gradients,
                )

        eval_dataset = Dataset({"system": eval_systems, **eval_targets})

        # Evaluate the model
        try:
            predictions = _eval_targets(
                model=model,
                dataset=eval_dataset,
                options=eval_info_dict,
                return_predictions=True,
            )
        except Exception as e:
            raise ArchitectureError(e)

        write_predictions(
            filename=f"{output.stem}{file_index_suffix}{output.suffix}",
            systems=eval_systems,
            capabilities=model.capabilities(),
            predictions=predictions,
        )
