from copy import deepcopy
from gymportal.environment.normalization import min_max_normalization
from torch.utils.data import Dataset
import torch
import pandas as pd


class MyDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, parquet_file, transform=None):
        """
        Arguments:
            parquet_file (string): Path to the parquet file with trajectories (we use this instead of csv because it works with numpy arrays natively).
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_parquet(parquet_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        landmarks = self.df.iloc[idx]

        sample = {key: value for key, value in landmarks.to_dict().items()}

        if self.transform:
            sample = self.transform(sample)

        return sample


class TransformAction(object):
    """Rescales the actions from [0, 32] to the interval [-1, 1]

    Args:

    """

    def __call__(self, sample):
        sample = deepcopy(sample)
        sample["action"] = min_max_normalization(
            old_min=0, old_max=32, new_min=-1, new_max=1, values=sample["action"])
        return sample
