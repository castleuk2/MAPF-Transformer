import glob
import os
import time

import numpy as np
import pyarrow as pa
import torch

from loguru import logger
from torch.utils.data import Dataset


class MapfArrowDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, device, batch_size):
        self.all_data_files = self.file_paths = sorted(glob.glob(os.path.join(folder_path, "*.arrow")))
        self.device = device
        self.batch_size = batch_size
        self.dtype = torch.int8

        ddp_local_rank = os.environ.get("LOCAL_RANK")
        ddp_world_size = os.environ.get("WORLD_SIZE")
        # Divide files among DDP workers for training
        if "train" in folder_path and ddp_local_rank is not None and ddp_world_size is not None:
            ddp_local_rank, ddp_world_size = int(ddp_local_rank), int(ddp_world_size)
            assignments = [[] for _ in range(ddp_world_size)]
            assigned_rows = [0 for _ in range(ddp_world_size)]
            for path in self.file_paths:
                rank = min(range(ddp_world_size), key=assigned_rows.__getitem__)
                assignments[rank].append(path)
                assigned_rows[rank] += self._row_count(path)
            self.file_paths = assignments[ddp_local_rank]

        if not self.file_paths:
            raise ValueError(f"No Arrow shards assigned from {folder_path}")
        self.full_dataset_size = sum(self._row_count(path) for path in self.all_data_files)
        self.shard_dataset_size = sum(self._row_count(path) for path in self.file_paths)
        self.input_tensors = None
        self.target_tensors = None

    @staticmethod
    def _row_count(file_path):
        with pa.memory_map(file_path) as source:
            return pa.ipc.open_file(source).read_all().num_rows

    @staticmethod
    def _get_data_from_file(file_path):
        with pa.memory_map(file_path) as source:
            table = pa.ipc.open_file(source).read_all()
            input_tensors = table["input_tensors"].to_numpy(zero_copy_only=False)
            gt_actions = table["gt_actions"].to_numpy(zero_copy_only=False)

        # shuffle data within the current file
        indices = np.random.permutation(len(input_tensors))
        input_tensors = np.stack(input_tensors[indices])
        gt_actions = gt_actions[indices]

        return input_tensors, gt_actions

    def load_and_transfer_data_file(self, filename):
        start_time = time.monotonic()

        input_tensors, gt_actions = self._get_data_from_file(filename)

        self.input_tensors = torch.as_tensor(input_tensors, dtype=self.dtype, device=self.device)
        self.target_tensors = torch.full(input_tensors.shape, -1, dtype=self.dtype, device=self.device)
        self.target_tensors[:, -1] = torch.as_tensor(gt_actions, dtype=self.dtype, device=self.device)
        finish_time = time.monotonic() - start_time
        logger.debug(f'Data from {filename} for {self.device} device prepared in ~{round(finish_time, 5)}s')

    def __iter__(self):
        while True:
            for file_path in self.file_paths:
                self.load_and_transfer_data_file(file_path)
                for i in range(0, len(self.input_tensors), self.batch_size):
                    yield self.input_tensors[i:i + self.batch_size], self.target_tensors[i:i + self.batch_size]

    def get_shard_size(self):
        return self.shard_dataset_size

    def get_full_dataset_size(self):
        return self.full_dataset_size


def main():
    # folder_path = "../dataset/validation"
    folder_path = "../dataset/train"
    dataset = MapfArrowDataset(folder_path, device='cuda:0', batch_size=32)
    data = iter(dataset)
    x = 0
    logger.info(dataset.get_full_dataset_size())
    logger.info(dataset.get_shard_size())

    while True:
        x += 1
        qx, qy = next(data)
        # logger.info(str(qx.shape) + ' ' + str(qy.shape))


if __name__ == "__main__":
    main()
