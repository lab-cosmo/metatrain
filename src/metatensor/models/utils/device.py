from typing import List

import torch


cpu_options = ["cpu"]
cuda_options = ["cuda", "gpu"]
multi_gpu_options = [
    "multiple_gpu",
    "multiple-gpu",
    "multi_gpu",
    "multi-gpu",
]
all_options = cpu_options + cuda_options + multi_gpu_options


def string_to_device(string: str) -> List[torch.device]:
    """Converts a string to a list of torch devices.

    This function is used to convert a user-provided ``device`` string
    into a list of ``torch.device`` objects, which can then be passed
    to ``metatensor-models`` functions.

    :param string: The string to convert.

    :return: A list of torch devices.
    """

    if string.lower() in cpu_options:
        return [torch.device("cpu")]

    if string.lower() in cuda_options:
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA is not available on this system, "
                "so the `cuda` option is not available."
            )
        return [torch.device("cuda")]

    if string.lower() in multi_gpu_options:
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise ValueError(
                "No CUDA-capable GPUs were found on this system, "
                "so the `multi-gpu` option is not available."
            )
        if device_count == 1:
            raise ValueError(
                "Only one CUDA-capable GPU was found on this system, "
                "so the `multi-gpu` option is not available."
            )
        return [torch.device(f"cuda:{i}") for i in range(device_count)]

    raise ValueError(
        f"Unrecognized device string `{string}`. " f"Valid options are: {all_options}"
    )
