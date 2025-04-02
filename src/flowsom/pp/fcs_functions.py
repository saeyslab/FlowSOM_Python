from __future__ import annotations

import random

import anndata as ad
import numpy as np

from flowsom.io import read_FCS
from flowsom.tl import get_markers


def aggregate_flowframes(filenames, c_total, channels=None, keep_order=False):
    """Aggregate multiple FCS files together.

    :param filenames: An array containing full paths to the FCS files
    :type filenames: np.array
    :param c_total: Total number of cells to write to the output file
    :type c_total: int
    :param channels: Channels/markers to keep in the aggregate. Default None
    takes all channels of the first file
    :type channels: np.array
    :param keep_order: If True, the random subsample will be ordered in the same
    way as they were originally ordered in the file. Default=False.
    :type keep_order: boolean
    :param silent: If False, prints an update every time it starts processing a
    new file. Default = False.
    :type silent: boolean.
    """
    nFiles = len(filenames)
    cFile = int(np.ceil(c_total / nFiles))

    flow_frame = []
    for i, file_path in enumerate(filenames):
        f = read_FCS(file_path)
        if channels is not None:
            f = f[:, list(get_markers(f, channels).keys())]
        cPerFile = min([f.X.shape[0], cFile])

        # Random sampling
        ids = random.sample(range(f.X.shape[0]), cPerFile)
        if keep_order:
            ids = sorted(ids)

        file_ids = np.repeat(i, cPerFile)

        f = f[ids,]
        f.obs["Original_ID"] = np.array(ids, dtype=np.float32)
        f.obs["File"] = np.array(file_ids, dtype=np.float32)
        f.obs["File_scattered"] = np.array(
            np.add(file_ids, np.random.normal(loc=0.0, scale=0.1, size=len(file_ids))), dtype=np.float32
        )
        flow_frame.append(f)
    flow_frame = ad.concat(flow_frame, join="outer", uns_merge="first")
    return flow_frame
