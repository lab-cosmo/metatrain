from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Union

import numpy as np
import torch
from metatensor.torch import TensorMap
from metatomic.torch import ModelCapabilities, System

from .writers import Writer, _split_tensormaps


class MemmapWriter(Writer):
    """
    Write systems and predictions to the on-disk layout consumed by
    :py:class:`metatrain.utils.data.dataset.MemmapDataset`: a directory containing
    ``ns.npy``, ``na.npy``, ``pbc.npy``, ``x.bin``, ``a.bin``, ``c.bin``, and one
    ``<target>.bin`` file per target (plus
    ``<target>_forces.bin``/``<target>_stress.bin`` for position/strain gradients). The
    resulting directory can be passed straight back to ``metatrain`` as a ``systems:
    read_from:`` dataset, which makes this format well suited for very large evaluation
    runs (e.g. >1M structures) that should not be re-read as a single in-memory XYZ
    file.

    The final number of structures/atoms is only known once evaluation has finished,
    so every array is streamed to its ``.bin`` file (append-only, on every ``write()``
    call) rather than buffered in memory or in a temporary directory. This keeps
    memory use and disk I/O to a single pass, at the cost of not knowing final shapes
    until ``finish()``, where the small ``ns.npy``/``na.npy``/``pbc.npy`` files (which
    require the final structure/atom counts) are written and all files are closed.

    :param path: Directory path to write into (e.g. ``predictions/``), used as-is
        (unlike the other writers, no filename manipulation is performed). The
        directory is created if it does not exist yet. If it already exists and
        already contains any file, a :py:class:`FileExistsError` is raised
        immediately rather than writing into (or overwriting) a possibly unrelated
        directory: this check happens up front, before any (potentially very long)
        evaluation runs, instead of lazily discovering a conflict only when the
        colliding file would have been written.
    :param capabilities: Model capabilities (unused, but matches base signature).
    :param append: Not supported for memmap datasets.
    """

    def __init__(
        self,
        path: Union[str, Path],
        capabilities: Optional[ModelCapabilities] = None,
        append: Optional[bool] = False,
    ):
        if append:
            raise ValueError("Appending is not supported for the memmap writer.")
        super().__init__(filename=path, capabilities=capabilities, append=append)

        self.directory = Path(path)
        if self.directory.exists() and any(self.directory.iterdir()):
            raise FileExistsError(
                f"'{self.directory}' already exists and is not empty. Refusing to "
                "write a memmap dataset into a non-empty directory; remove its "
                "contents first, or point to a different directory."
            )
        self.directory.mkdir(parents=True, exist_ok=True)

        self._atom_counts: List[int] = []
        self._pbc: List[np.ndarray] = []
        self._files: Dict[str, BinaryIO] = {}

    def _file(self, name: str) -> BinaryIO:
        if name not in self._files:
            self._files[name] = open(self.directory / name, "wb")
        return self._files[name]

    def write(self, systems: List[System], predictions: Dict[str, TensorMap]) -> None:
        """
        Append a batch of systems and predictions to the memmap dataset files.

        :param systems: List of systems to write.
        :param predictions: Dictionary of TensorMaps with predictions for the systems.
        """
        cpu_systems = [system.to("cpu").to(torch.float64) for system in systems]
        per_system_preds = _split_tensormaps(cpu_systems, predictions)

        for system, system_predictions in zip(
            cpu_systems, per_system_preds, strict=True
        ):
            self._atom_counts.append(len(system))
            self._pbc.append(system.pbc.numpy())
            system.types.to(torch.int32).numpy().tofile(self._file("a.bin"))
            system.positions.to(torch.float32).numpy().tofile(self._file("x.bin"))
            system.cell.to(torch.float32).numpy().tofile(self._file("c.bin"))

            for target_name, tensor_map in system_predictions.items():
                if len(tensor_map.keys) != 1:
                    raise ValueError(
                        "Only single-block `TensorMap`s can be written to memmap "
                        "datasets for the moment."
                    )
                block = tensor_map.block()
                values = block.values.detach().cpu().to(torch.float32)
                values.numpy().tofile(self._file(f"{target_name}.bin"))

                for gradient_name, gradient_block in block.gradients():
                    gvalues = gradient_block.values.detach().cpu().to(torch.float32)
                    if gradient_name == "positions":
                        # force = -dE/dx
                        gvalues = -gvalues
                        key = f"{target_name}_forces"
                    elif gradient_name == "strain":
                        # MemmapDataset only supports stresses, not virials. It
                        # reconstructs the strain gradient as stress * |det(cell)|
                        # (see MemmapDataset.__getitem__), so the volume used here
                        # must be the absolute value to round-trip correctly for
                        # left-handed (negative-determinant) cells too.
                        volume = torch.abs(torch.det(system.cell)).item()
                        if volume == 0:
                            gvalues = torch.full_like(gvalues, float("nan"))
                        else:
                            gvalues = gvalues / volume
                        key = f"{target_name}_stress"
                    else:
                        continue
                    gvalues.numpy().tofile(self._file(f"{key}.bin"))

    def finish(self) -> None:
        """Write ``ns.npy``/``na.npy``/``pbc.npy`` and close all open binary files."""
        ns = len(self._atom_counts)
        na: np.ndarray = np.cumsum(np.array([0] + self._atom_counts, dtype=np.int64))
        np.save(self.directory / "ns.npy", ns)
        np.save(self.directory / "na.npy", na)
        np.save(
            self.directory / "pbc.npy",
            np.array(self._pbc, dtype=bool).reshape(ns, 3),
        )

        for f in self._files.values():
            f.close()
        self._files = {}
