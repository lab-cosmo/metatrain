import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.distributed
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput
from torch.utils.data import DataLoader, DistributedSampler

from ...utils.composition import calculate_composition_weights
from ...utils.data import (
    CombinedDataLoader,
    Dataset,
    DatasetInfo,
    check_datasets,
    collate_fn,
    get_all_species,
    get_all_targets,
)
from ...utils.distributed.distributed_data_parallel import DistributedDataParallel
from ...utils.distributed.slurm import DistributedEnvironment
from ...utils.evaluate_model import evaluate_model
from ...utils.extract_targets import get_outputs_dict
from ...utils.io import is_exported, load, save
from ...utils.logging import MetricLogger
from ...utils.loss import TensorMapDictLoss
from ...utils.merge_capabilities import merge_capabilities
from ...utils.metrics import RMSEAccumulator
from ...utils.per_atom import divide_by_num_atoms
from . import DEFAULT_HYPERS
from .model import Model


logger = logging.getLogger(__name__)


# Filter out the second derivative and device warnings from rascaline-torch
warnings.filterwarnings("ignore", category=UserWarning, message="second derivative")
warnings.filterwarnings(
    "ignore", category=UserWarning, message="Systems data is on device"
)


def train(
    train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
    validation_datasets: List[Union[Dataset, torch.utils.data.Subset]],
    dataset_info: DatasetInfo,
    devices: List[torch.device],
    hypers: Dict = DEFAULT_HYPERS,
    continue_from: Optional[str] = None,
    checkpoint_dir: str = ".",
):

    is_distributed = hypers["training"]["distributed"]

    if is_distributed:
        distr_env = DistributedEnvironment()
        torch.distributed.init_process_group(backend="nccl")
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    all_species = get_all_species(train_datasets + validation_datasets)
    outputs = {
        key: ModelOutput(
            quantity=value.quantity,
            unit=value.unit,
            per_atom=True,
        )
        for key, value in dataset_info.targets.items()
    }

    dtype = train_datasets[0][0]["system"].positions.dtype
    if dtype == torch.float64:
        dtype_string = "float64"
    elif dtype == torch.float32:
        dtype_string = "float32"
    else:
        raise ValueError(f"Unsupported dtype {dtype} in SOAP-BPNN")

    # the model is always capable of outputting the last layer features
    outputs["mtm::aux::last_layer_features"] = ModelOutput(per_atom=True)

    new_capabilities = ModelCapabilities(
        length_unit=dataset_info.length_unit,
        outputs=outputs,
        atomic_types=all_species,
        supported_devices=["cuda", "cpu"],
        interaction_range=hypers["model"]["soap"]["cutoff"],
        dtype=dtype_string,
    )

    # Create the model:
    if continue_from is None:
        model = Model(
            capabilities=new_capabilities,
            hypers=hypers["model"],
        )
        novel_capabilities = new_capabilities
    else:
        model = load(continue_from)
        if is_exported(model):
            raise ValueError("model is already exported and can't be used for continue")

        filtered_new_dict = {k: v for k, v in hypers["model"].items() if k != "restart"}
        filtered_old_dict = {k: v for k, v in model.hypers.items() if k != "restart"}
        if filtered_new_dict != filtered_old_dict:
            logger.warning(
                "The hyperparameters of the model have changed since the last "
                "training run. The new hyperparameters will be discarded."
            )
        # merge the model's capabilities with the requested capabilities
        merged_capabilities, novel_capabilities = merge_capabilities(
            model.capabilities, new_capabilities
        )
        model.capabilities = merged_capabilities
        # make the new model capable of handling the new outputs
        for output_name in novel_capabilities.outputs.keys():
            model.add_output(output_name)

    model_capabilities = model.capabilities

    # Perform checks on the datasets:
    logger.info("Checking datasets for consistency")
    try:
        check_datasets(train_datasets, validation_datasets)
    except ValueError as err:
        if continue_from is not None:
            logger.warning(err)
        else:
            # only error if we are not continuing
            raise ValueError(err) from err

    if is_distributed:
        device = torch.device("cuda", distr_env.local_rank)
    else:
        device = devices[0]  # only one device, as we don't support multi-gpu for now
    dtype = train_datasets[0][0]["system"].positions.dtype

    if is_distributed:
        logger.info(f"Training on {world_size} devices with dtype {dtype}")
    else:
        logger.info(f"Training on device {device} with dtype {dtype}")
    model.to(device=device, dtype=dtype)
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[device])

    hypers_training = hypers["training"]

    # Calculate and set the composition weights for all targets:
    logger.info("Calculating composition weights")
    for target_name in novel_capabilities.outputs.keys():
        if "mtm::aux::" in target_name:
            continue
        # TODO: warn in the documentation that capabilities that are already
        # present in the model won't recalculate the composition weights
        # find the datasets that contain the target:

        if target_name in hypers_training["fixed_composition_weights"].keys():
            logger.info(
                f"For {target_name}, model will proceed with "
                "user-supplied composition weights"
            )
            cur_weight_dict = hypers_training["fixed_composition_weights"][target_name]
            species = []
            num_species = len(cur_weight_dict)
            fixed_weights = torch.zeros(num_species, dtype=dtype, device=device)

            for ii, (key, weight) in enumerate(cur_weight_dict.items()):
                species.append(key)
                fixed_weights[ii] = weight

            all_species = []
            for dataset in train_datasets:
                all_species += get_all_species(dataset)

            if not set(species) == set(all_species):
                raise ValueError(
                    "Values were not supplied for all "
                    "the species in present in the dataset"
                )
            (model.module if is_distributed else model).set_composition_weights(
                target_name, fixed_weights, species
            )

        else:
            train_datasets_with_target = []
            for dataset in train_datasets:
                if target_name in get_all_targets(dataset):
                    train_datasets_with_target.append(dataset)
            if len(train_datasets_with_target) == 0:
                raise ValueError(
                    f"Target {target_name} in the model's new capabilities is not "
                    "present in any of the training datasets."
                )
            composition_weights, species = calculate_composition_weights(
                train_datasets_with_target, target_name
            )
            (model.module if is_distributed else model).set_composition_weights(
                target_name, composition_weights, species
            )

    logger.info("Setting up data loaders")

    if is_distributed:
        train_samplers = [
            DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
            for train_dataset in train_datasets
        ]
        validation_samplers = [
            DistributedSampler(
                validation_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
            for validation_dataset in validation_datasets
        ]
    else:
        train_samplers = [None] * len(train_datasets)
        validation_samplers = [None] * len(validation_datasets)

    # Create dataloader for the training datasets:
    train_dataloaders = []
    for dataset, sampler in zip(train_datasets, train_samplers):
        train_dataloaders.append(
            DataLoader(
                dataset=dataset,
                batch_size=hypers_training["batch_size"],
                sampler=sampler,
                shuffle=(sampler is None),
                drop_last=(sampler is None),
                collate_fn=collate_fn,
            )
        )
    train_dataloader = CombinedDataLoader(train_dataloaders, shuffle=True)

    # Create dataloader for the validation datasets:
    validation_dataloaders = []
    for dataset, sampler in zip(validation_datasets, validation_samplers):
        validation_dataloaders.append(
            DataLoader(
                dataset=dataset,
                batch_size=hypers_training["batch_size"],
                sampler=sampler,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn,
            )
        )
    validation_dataloader = CombinedDataLoader(validation_dataloaders, shuffle=False)

    # Extract all the possible outputs and their gradients from the training set:
    outputs_dict = get_outputs_dict(train_datasets)
    for output_name in outputs_dict.keys():
        if output_name not in model_capabilities.outputs:
            raise ValueError(
                f"Output {output_name} is not in the model's capabilities."
            )

    # Create a loss weight dict:
    loss_weights_dict = {}
    for output_name, value_or_gradient_list in outputs_dict.items():
        loss_weights_dict[output_name] = {
            value_or_gradient: 1.0 for value_or_gradient in value_or_gradient_list
        }

    # Create a loss function:
    loss_fn = TensorMapDictLoss(loss_weights_dict)

    # Create an optimizer:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hypers_training["learning_rate"]
    )

    # Create a scheduler:
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=hypers_training["scheduler_factor"],
        patience=hypers_training["scheduler_patience"],
    )

    # counters for early stopping:
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    # per-atom targets:
    per_structure_targets = hypers_training["per_structure_targets"]

    # Train the model:
    logger.info("Starting training")
    for epoch in range(hypers_training["num_epochs"]):
        if is_distributed:
            torch.distributed.barrier()
        train_rmse_calculator = RMSEAccumulator()
        validation_rmse_calculator = RMSEAccumulator()

        train_loss = 0.0
        for batch in train_dataloader:
            optimizer.zero_grad()

            systems, targets = batch
            systems = [system.to(device=device) for system in systems]
            targets = {key: value.to(device=device) for key, value in targets.items()}
            predictions = evaluate_model(
                model,
                systems,
                {key: dataset_info.targets[key] for key in targets.keys()},
                is_training=True,
            )

            # average by the number of atoms
            predictions, targets = _average_by_num_atoms(
                predictions, targets, systems, per_structure_targets
            )

            train_loss_batch = loss_fn(predictions, targets)
            train_loss_batch.backward()
            optimizer.step()

            if is_distributed:
                torch.distributed.all_reduce(train_loss_batch)
            train_loss += train_loss_batch.item()
            train_rmse_calculator.update(predictions, targets)
        finalized_train_info = train_rmse_calculator.finalize(
            not_per_atom=["positions_gradients"] + per_structure_targets,
            is_distributed=is_distributed,
            device=device,
        )

        if is_distributed:
            torch.distributed.barrier()

        validation_loss = 0.0
        for batch in validation_dataloader:
            systems, targets = batch
            systems = [system.to(device=device) for system in systems]
            targets = {key: value.to(device=device) for key, value in targets.items()}
            predictions = evaluate_model(
                model,
                systems,
                {key: dataset_info.targets[key] for key in targets.keys()},
                is_training=False,
            )

            # average by the number of atoms
            predictions, targets = _average_by_num_atoms(
                predictions, targets, systems, per_structure_targets
            )

            validation_loss_batch = loss_fn(predictions, targets)

            if is_distributed:
                torch.distributed.all_reduce(validation_loss_batch)
            validation_loss += validation_loss_batch.item()
            validation_rmse_calculator.update(predictions, targets)
        finalized_validation_info = validation_rmse_calculator.finalize(
            not_per_atom=["positions_gradients"] + per_structure_targets,
            is_distributed=is_distributed,
            device=device,
        )

        lr_scheduler.step(validation_loss)

        # Now we log the information:
        finalized_train_info = {"loss": train_loss, **finalized_train_info}
        finalized_validation_info = {
            "loss": validation_loss,
            **finalized_validation_info,
        }

        if epoch == 0:
            metric_logger = MetricLogger(
                model_capabilities=model_capabilities,
                initial_metrics=[finalized_train_info, finalized_validation_info],
                names=["train", "validation"],
            )
        if epoch % hypers_training["log_interval"] == 0:
            metric_logger.log(
                metrics=[finalized_train_info, finalized_validation_info],
                epoch=epoch,
            )

        if epoch % hypers_training["checkpoint_interval"] == 0:
            if is_distributed:
                torch.distributed.barrier()
            save(
                (model.module if is_distributed else model),
                Path(checkpoint_dir) / f"model_{epoch}.ckpt",
            )

        # early stopping criterion:
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= hypers_training["early_stopping_patience"]:
                logger.info(
                    "Early stopping criterion reached after "
                    f"{hypers_training['early_stopping_patience']} epochs "
                    "without improvement."
                )
                break

    if is_distributed:
        torch.distributed.barrier()

    return model.module if is_distributed else model


def _average_by_num_atoms(predictions, targets, systems, per_structure_targets):
    device = systems[0].device
    num_atoms = torch.tensor([len(s) for s in systems], device=device)
    for target in targets.keys():
        if target in per_structure_targets:
            continue
        predictions[target] = divide_by_num_atoms(predictions[target], num_atoms)
        targets[target] = divide_by_num_atoms(targets[target], num_atoms)

    return predictions, targets
