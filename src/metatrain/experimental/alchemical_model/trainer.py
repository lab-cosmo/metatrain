import logging
from pathlib import Path
from typing import List, Union

import torch
from metatensor.learn.data import DataLoader

from ...utils.composition import calculate_composition_weights
from ...utils.data import (
    CombinedDataLoader,
    Dataset,
    TargetInfoDict,
    check_datasets,
    collate_fn,
    get_all_targets,
)
from ...utils.evaluate_model import evaluate_model
from ...utils.external_naming import to_external_name
from ...utils.io import check_suffix
from ...utils.logging import MetricLogger
from ...utils.loss import TensorMapDictLoss
from ...utils.metrics import RMSEAccumulator
from ...utils.neighbor_lists import get_system_with_neighbor_lists
from ...utils.per_atom import average_by_num_atoms
from . import AlchemicalModel
from .utils.normalize import (
    get_average_number_of_atoms,
    get_average_number_of_neighbors,
    remove_composition_from_dataset,
)


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, train_hypers):
        self.hypers = train_hypers
        self.optimizer_state_dict = None
        self.scheduler_state_dict = None
        self.epoch = None

    def train(
        self,
        model: AlchemicalModel,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ):
        dtype = train_datasets[0][0]["system"].positions.dtype
        device = devices[0]  # only one device, as we don't support multi-gpu for now

        if len(model.dataset_info.targets) != 1:
            raise ValueError("The Alchemical Model only supports a single target")
        target_name = next(iter(model.dataset_info.targets.keys()))
        if model.dataset_info.targets[target_name].quantity != "energy":
            raise ValueError("The Alchemical Model only supports energies as target")
        if model.dataset_info.targets[target_name].per_atom:
            raise ValueError("The Alchemical Model does not support per-atom training")

        # Perform canonical checks on the datasets:
        logger.info("Checking datasets for consistency")
        check_datasets(train_datasets, val_datasets)

        # Calculating the neighbor lists for the training and validation datasets:
        logger.info("Calculating neighbor lists for the datasets")
        requested_neighbor_lists = model.requested_neighbor_lists()
        for dataset in train_datasets + val_datasets:
            for i in range(len(dataset)):
                system = dataset[i]["system"]
                # The following line attaches the neighbors lists to the system,
                # and doesn't require to reassign the system to the dataset:
                _ = get_system_with_neighbor_lists(system, requested_neighbor_lists)

        # Calculate the average number of atoms and neighbor in the training datasets:
        average_number_of_atoms = get_average_number_of_atoms(train_datasets)
        average_number_of_neighbors = get_average_number_of_neighbors(train_datasets)

        # Given that currently multiple datasets are not supported, we can assume that:
        average_number_of_atoms = average_number_of_atoms[0]
        average_number_of_neighbors = average_number_of_neighbors[0]

        # Set the normalization factors for the model:
        model.set_normalization_factor(average_number_of_atoms)
        model.set_basis_normalization_factor(average_number_of_neighbors)

        logger.info(f"Training on device {device} with dtype {dtype}")
        model.to(device=device, dtype=dtype)

        # Calculate and set the composition weights, but only if
        # this is the first training run:
        if not model.is_restarted:
            for target_name in model.outputs.keys():
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
                model.set_composition_weights(
                    composition_weights.unsqueeze(0), composition_types
                )

        # Remove the composition from the datasets:
        train_datasets = [
            remove_composition_from_dataset(
                train_datasets[0],
                model.atomic_types,
                model.alchemical_model.composition_weights.squeeze(0),
            )
        ]
        val_datasets = [
            remove_composition_from_dataset(
                val_datasets[0],
                model.atomic_types,
                model.alchemical_model.composition_weights.squeeze(0),
            )
        ]

        logger.info("Setting up data loaders")

        # Create dataloader for the training datasets:
        train_dataloaders = []
        for dataset in train_datasets:
            train_dataloaders.append(
                DataLoader(
                    dataset=dataset,
                    batch_size=self.hypers["batch_size"],
                    shuffle=True,
                    collate_fn=collate_fn,
                )
            )
        train_dataloader = CombinedDataLoader(train_dataloaders, shuffle=True)

        # Create dataloader for the validation datasets:
        val_dataloaders = []
        for dataset in val_datasets:
            val_dataloaders.append(
                DataLoader(
                    dataset=dataset,
                    batch_size=self.hypers["batch_size"],
                    shuffle=False,
                    collate_fn=collate_fn,
                )
            )
        val_dataloader = CombinedDataLoader(val_dataloaders, shuffle=False)

        # Extract all the possible outputs and their gradients:
        outputs_list = []
        for target_name, target_info in model.dataset_info.targets.items():
            outputs_list.append(target_name)
            for gradient_name in target_info.gradients:
                outputs_list.append(f"{target_name}_{gradient_name}_gradients")
        # Create a loss weight dict:
        loss_weights_dict = {}
        for output_name in outputs_list:
            loss_weights_dict[output_name] = (
                self.hypers["loss_weights"][
                    to_external_name(output_name, model.outputs)
                ]
                if to_external_name(output_name, model.outputs)
                in self.hypers["loss_weights"]
                else 1.0
            )
        loss_weights_dict_external = {
            to_external_name(key, model.outputs): value
            for key, value in loss_weights_dict.items()
        }
        logging.info(f"Training with loss weights: {loss_weights_dict_external}")

        # Create a loss function:
        loss_fn = TensorMapDictLoss(loss_weights_dict)

        # Create an optimizer:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.hypers["learning_rate"]
        )
        if self.optimizer_state_dict is not None:
            optimizer.load_state_dict(self.optimizer_state_dict)

        # Create a scheduler:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.hypers["scheduler_factor"],
            patience=self.hypers["scheduler_patience"],
        )
        if self.scheduler_state_dict is not None:
            lr_scheduler.load_state_dict(self.scheduler_state_dict)

        # counters for early stopping:
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        # per-atom targets:
        per_structure_targets = self.hypers["per_structure_targets"]

        start_epoch = 0 if self.epoch is None else self.epoch + 1

        # Train the model:
        logger.info("Starting training")
        for epoch in range(start_epoch, start_epoch + self.hypers["num_epochs"]):
            train_rmse_calculator = RMSEAccumulator()
            val_rmse_calculator = RMSEAccumulator()

            train_loss = 0.0
            for batch in train_dataloader:
                optimizer.zero_grad()

                systems, targets = batch
                assert len(systems[0].known_neighbor_lists()) > 0
                systems = [system.to(device=device) for system in systems]
                targets = {
                    key: value.to(device=device) for key, value in targets.items()
                }
                predictions = evaluate_model(
                    model,
                    systems,
                    TargetInfoDict(
                        **{
                            key: model.dataset_info.targets[key]
                            for key in targets.keys()
                        }
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
                train_rmse_calculator.update(predictions, targets)
            finalized_train_info = train_rmse_calculator.finalize(
                not_per_atom=["positions_gradients"] + per_structure_targets
            )

            val_loss = 0.0
            for batch in val_dataloader:
                systems, targets = batch
                assert len(systems[0].known_neighbor_lists()) > 0
                systems = [system.to(device=device) for system in systems]
                targets = {
                    key: value.to(device=device) for key, value in targets.items()
                }
                predictions = evaluate_model(
                    model,
                    systems,
                    TargetInfoDict(
                        **{
                            key: model.dataset_info.targets[key]
                            for key in targets.keys()
                        }
                    ),
                    is_training=False,
                )

                # average by the number of atoms
                predictions = average_by_num_atoms(
                    predictions, systems, per_structure_targets
                )
                targets = average_by_num_atoms(targets, systems, per_structure_targets)

                val_loss_batch = loss_fn(predictions, targets)
                val_loss += val_loss_batch.item()
                val_rmse_calculator.update(predictions, targets)
            finalized_val_info = val_rmse_calculator.finalize(
                not_per_atom=["positions_gradients"] + per_structure_targets
            )

            lr_scheduler.step(val_loss)

            # Now we log the information:
            finalized_train_info = {"loss": train_loss, **finalized_train_info}
            finalized_val_info = {
                "loss": val_loss,
                **finalized_val_info,
            }

            if epoch == start_epoch:
                metric_logger = MetricLogger(
                    log_obj=logger,
                    dataset_info=model.dataset_info,
                    initial_metrics=[finalized_train_info, finalized_val_info],
                    names=["training", "validation"],
                )
            if epoch % self.hypers["log_interval"] == 0:
                metric_logger.log(
                    metrics=[finalized_train_info, finalized_val_info],
                    epoch=epoch,
                )

            if epoch % self.hypers["checkpoint_interval"] == 0:
                self.optimizer_state_dict = optimizer.state_dict()
                self.scheduler_state_dict = lr_scheduler.state_dict()
                self.epoch = epoch
                self.save_checkpoint(
                    model, Path(checkpoint_dir) / f"model_{epoch}.ckpt"
                )

            # early stopping criterion:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
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

    def save_checkpoint(self, model, path: Union[str, Path]):
        checkpoint = {
            "model_hypers": {
                "model_hypers": model.hypers,
                "dataset_info": model.dataset_info,
            },
            "model_state_dict": model.state_dict(),
            "train_hypers": self.hypers,
            "epoch": self.epoch,
            "optimizer_state_dict": self.optimizer_state_dict,
            "scheduler_state_dict": self.scheduler_state_dict,
        }
        torch.save(
            checkpoint,
            check_suffix(path, ".ckpt"),
        )

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path], train_hypers) -> "Trainer":

        # Load the checkpoint
        checkpoint = torch.load(path)
        model_hypers = checkpoint["model_hypers"]
        model_state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint["epoch"]
        optimizer_state_dict = checkpoint["optimizer_state_dict"]
        scheduler_state_dict = checkpoint["scheduler_state_dict"]

        # Create the trainer
        trainer = cls(train_hypers)
        trainer.optimizer_state_dict = optimizer_state_dict
        trainer.scheduler_state_dict = scheduler_state_dict
        trainer.epoch = epoch

        # Create the model
        model = AlchemicalModel(**model_hypers)
        model.load_state_dict(model_state_dict)

        return trainer
