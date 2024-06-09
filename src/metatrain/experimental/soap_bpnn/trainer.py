import logging
import warnings
from pathlib import Path
from typing import List, Union

import torch
import torch.distributed
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput
from torch.utils.data import DataLoader, DistributedSampler

from ...utils.composition import calculate_composition_weights
from ...utils.data import (
    CombinedDataLoader,
    Dataset,
    TargetInfoDict,
    collate_fn,
    get_all_targets,
)
from ...utils.data.extract_targets import get_targets_dict
from ...utils.evaluate_model import evaluate_model
from ...utils.external_naming import to_external_name
from ...utils.logging import MetricLogger
from ...utils.loss import TensorMapDictLoss
from ...utils.metrics import RMSEAccumulator
from ...utils.per_atom import average_by_num_atoms
from .model import SoapBpnn
from ...utils.distributed.distributed_data_parallel import DistributedDataParallel
from ...utils.distributed.slurm import DistributedEnvironment


logger = logging.getLogger(__name__)


# Filter out the second derivative and device warnings from rascaline-torch
warnings.filterwarnings("ignore", category=UserWarning, message="second derivative")
warnings.filterwarnings(
    "ignore", category=UserWarning, message="Systems data is on device"
)


class Trainer:
    def __init__(self, train_hypers):
        self.hypers = train_hypers

    def train(
        self,
        model: SoapBpnn,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        validation_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ):

        is_distributed = hypers["training"]["distributed"]

        if is_distributed:
            distr_env = DistributedEnvironment(hypers["training"]["distributed_port"])
            torch.distributed.init_process_group(backend="nccl")
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        if is_distributed:
            if len(devices) > 1:
                raise ValueError(
                    "Requested distributed training with the `multi-gpu` device. If you "
                    "want to run distributed training with SOAP-BPNN, please set `device` "
                    "to cuda."
                )
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

        # Calculate and set the composition weights for all targets:
        logger.info("Calculating composition weights")
        for target_name in model.new_outputs:
            if "mtm::aux::" in target_name:
                continue
            # TODO: document transfer learning and say that outputs that are already
            # present in the model will keep their composition weights
            if target_name in self.hypers["fixed_composition_weights"].keys():
                logger.info(
                    f"For {target_name}, model will use "
                    "user-supplied composition weights"
                )
                cur_weight_dict = self.hypers["fixed_composition_weights"][target_name]
                atomic_types = set()
                num_species = len(cur_weight_dict)
                fixed_weights = torch.zeros(num_species, dtype=dtype, device=device)

                for ii, (key, weight) in enumerate(cur_weight_dict.items()):
                    atomic_types.add(key)
                    fixed_weights[ii] = weight

                if not set(atomic_types) == model.atomic_types:
                    raise ValueError(
                        "Supplied atomic types are not present in the dataset."
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
                composition_weights, composition_types = calculate_composition_weights(
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

        # Extract all the possible outputs and their gradients:
        training_targets = get_targets_dict(train_datasets, model.dataset_info)
        outputs_list = []
        for target_name, target_info in training_targets.items():
            outputs_list.append(target_name)
            for gradient_name in target_info.gradients:
                outputs_list.append(f"{target_name}_{gradient_name}_gradients")
        # Create a loss weight dict:
        loss_weights_dict = {}
        for output_name in outputs_list:
            loss_weights_dict[output_name] = (
                self.hypers["loss_weights"][
                    to_external_name(output_name, training_targets)
                ]
                if to_external_name(output_name, training_targets)
                in self.hypers["loss_weights"]
                else 1.0
            )
        loss_weights_dict_external = {
            to_external_name(key, training_targets): value
            for key, value in loss_weights_dict.items()
        }
        logging.info(f"Training with loss weights: {loss_weights_dict_external}")

        # Create a loss function:
        loss_fn = TensorMapDictLoss(loss_weights_dict)

        # Create an optimizer:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.hypers["learning_rate"]
        )

        # Create a scheduler:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.hypers["scheduler_factor"],
            patience=self.hypers["scheduler_patience"],
        )

        # counters for early stopping:
        best_validation_loss = float("inf")
        epochs_without_improvement = 0

        # per-atom targets:
        per_structure_targets = self.hypers["per_structure_targets"]

        # Train the model:
        logger.info("Starting training")
        for epoch in range(self.hypers["num_epochs"]):
            train_rmse_calculator = RMSEAccumulator()
            validation_rmse_calculator = RMSEAccumulator()

            train_loss = 0.0
            for batch in train_dataloader:
                optimizer.zero_grad()

                systems, targets = batch
                systems = [system.to(device=device) for system in systems]
                targets = {
                    key: value.to(device=device) for key, value in targets.items()
                }
                predictions = evaluate_model(
                    model,
                    systems,
                    TargetInfoDict(
                        **{key: training_targets[key] for key in targets.keys()}
                    ),
                    is_training=True,
                )

                # average by the number of atoms
                predictions = average_by_num_atoms(
                    predictions, systems, per_structure_targets
                )
                targets = average_by_num_atoms(targets, systems, per_structure_targets)

                train_loss_batch = loss_fn(predictions, targets)
                train_loss += train_loss_batch.item()
                train_loss_batch.backward()
                optimizer.step()

                if is_distributed:
                    # sum the loss over all processes
                    torch.distributed.all_reduce(train_loss_batch)
                train_loss += train_loss_batch.item()
                train_rmse_calculator.update(predictions, targets)
            finalized_train_info = train_rmse_calculator.finalize(
                not_per_atom=["positions_gradients"] + per_structure_targets,
                is_distributed=is_distributed,
                device=device,
            )

            validation_loss = 0.0
            for batch in validation_dataloader:
                systems, targets = batch
                systems = [system.to(device=device) for system in systems]
                targets = {
                    key: value.to(device=device) for key, value in targets.items()
                }
                predictions = evaluate_model(
                    model,
                    systems,
                    TargetInfoDict(
                        **{key: training_targets[key] for key in targets.keys()}
                    ),
                    is_training=False,
                )

                # average by the number of atoms
                predictions = average_by_num_atoms(
                    predictions, systems, per_structure_targets
                )
                targets = average_by_num_atoms(targets, systems, per_structure_targets)

                validation_loss_batch = loss_fn(predictions, targets)

                if is_distributed:
                    # sum the loss over all processes
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
                    logobj=logger,
                    model_outputs=model.outputs,
                    initial_metrics=[finalized_train_info, finalized_validation_info],
                    names=["train", "validation"],
                )
            if epoch % self.hypers["log_interval"] == 0:
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
                if epochs_without_improvement >= self.hypers["early_stopping_patience"]:
                    logger.info(
                        "Early stopping criterion reached after "
                        f"{self.hypers['early_stopping_patience']} epochs "
                        "without improvement."
                    )
                    break
