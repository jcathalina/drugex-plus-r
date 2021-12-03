import numpy as np
import torch


def unique(arr):
    """Finds unique rows in arr and return their indices"""
    if type(arr) == torch.Tensor:
        arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(
        np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))
    )
    _, idxs = np.unique(arr_, return_index=True)
    idxs = np.sort(idxs)
    if type(arr) == torch.Tensor:
        idxs = torch.LongTensor(idxs).to(arr.get_device())
    return idxs
