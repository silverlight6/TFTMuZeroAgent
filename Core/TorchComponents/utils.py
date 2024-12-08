from typing import Dict, Optional
import torch
import numpy as np
import tree  # pip install dm_tree

def convert_to_torch_tensor(x, device: Optional[str] = None):
    """Converts any struct to torch.Tensors.

    x: Any (possibly nested) struct, the values in which will be
        converted and returned as a new struct with all leaves converted
        to torch tensors.

    Returns:
        Any: A new struct with the same structure as `x`, but with all
            values converted to torch Tensor types. This does not convert possibly
            nested elements that are None because torch has no representation for that.
    """

    def mapping(item):
        if item is None:
            # Torch has no representation for `None`, so we return None
            return item

        # Already torch tensor -> make sure it's on right device.
        if torch.is_tensor(item):
            tensor = item
        # Numpy arrays.
        elif isinstance(item, np.ndarray):
            # Object type (e.g. info dicts in train batch): leave as-is.
            # str type (e.g. agent_id in train batch): leave as-is.
            if item.dtype == object or item.dtype.type is np.str_:
                return item

            # Already numpy: Wrap as torch tensor.
            else:
                tensor = torch.from_numpy(item)
        # Everything else: Convert to numpy, then wrap as torch tensor.
        else:
            tensor = torch.from_numpy(np.asarray(item))

        # Floatify all float64 tensors.
        if tensor.is_floating_point():
            tensor = tensor.float()

        return tensor if device is None else tensor.to(device)

    return tree.map_structure(mapping, x)
