import warnings
from pathlib import Path
from typing import Any, Union
from urllib.parse import urlparse
from urllib.request import urlretrieve

import torch
from metatensor.torch.atomistic import check_atomistic_model

from .architectures import import_architecture


def check_file_extension(
    filename: Union[str, Path], extension: str
) -> Union[str, Path]:
    """Check the file extension of a file name and adds if it is not present.

    If ``filename`` does not end with ``extension`` the ``extension`` is added and a
    warning will be issued.

    :param filename: Name of the file to be checked.
    :param extension: Expected file extension i.e. ``.txt``.
    :returns: Checked and probably extended file name.
    """
    path_filename = Path(filename)

    if path_filename.suffix != extension:
        warnings.warn(
            f"The file name should have a '{extension}' file extension. The user "
            f"requested the file with name '{filename}', but it will be saved as "
            f"'{filename}{extension}'.",
            stacklevel=1,
        )
        path_filename = path_filename.parent / (path_filename.name + extension)

    if type(filename) is str:
        return str(path_filename)
    else:
        return path_filename


def is_exported_file(path: str) -> bool:
    """Check if a saved model file has been exported to a MetatensorAtomisticModel.

    :param path: model path
    :return: :py:obj:`True` if the ``model`` has been exported, :py:obj:`False`
        otherwise.

    .. seealso::
        :py:func:`utils.export.is_exported <metatrain.utils.export.is_exported>` to
        verify if an already loaded model is exported.
    """
    try:
        check_atomistic_model(str(path))
        return True
    except ValueError:
        return False


def load_model(architecture_name: str, path: Union[str, Path]) -> Any:
    """Loads a module from an URL or a local file.

    :param name: name of the architecture
    :param path: local or remote path to a model. For supported URL schemes see
        :py:class`urllib.request`
    :raises ValueError: if ``path`` is a YAML option file and no model
    :raises ValueError: if the checkpoint saved in ``path`` does not math the given
        ``architecture_name``
    """
    if Path(path).suffix in [".yaml", ".yml"]:
        raise ValueError(f"path '{path}' seems to be a YAML option file and no model")

    if urlparse(str(path)).scheme:
        path, _ = urlretrieve(str(path))

    if is_exported_file(str(path)):
        return torch.jit.load(str(path))
    else:  # model is a checkpoint
        architecture = import_architecture(architecture_name)

        try:
            return architecture.__model__.load_checkpoint(str(path))
        except Exception as err:
            raise ValueError(
                f"path '{path}' is not a valid model file for the {architecture_name} "
                "architecture"
            ) from err
