from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import torch
from torch_geometric.data import OnDiskDataset
from torch_geometric.data.data import BaseData
from torch_geometric.data import Data
from torch_geometric.io import fs
from tqdm import tqdm


class FensOnDisk(OnDiskDataset):
    """Preprocess FENDataset (csv) to graphs (pt) on disk

    .. note::
        This dataset uses the :class:`OnDiskDataset` base class to load data
        dynamically from disk.

    Args:
        root (str): Root directory where the dataset should be saved.
        split (str, optional): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: None)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        backend (str): The :class:`Database` backend to use.
            (default: :obj:`"sqlite"`)
        from_fens (callable, optional): A custom function that takes a FEN
            string and outputs a :obj:`~torch_geometric.data.Data` object.


    """

    def __init__(self, root, split=None, transform=None, pre_transform=None, from_fens: Optional[Callable] = None, create_edges: Optional[Callable] = None):

        self.from_fens = from_fens
        self.edges = create_edges()
        self.split = split

        schema = {
            'x': dict(dtype=torch.int64, size=(-1, 13)),
            'edge_index': dict(dtype=torch.int64, size=(2, -1)),
            'edge_attr': dict(dtype=torch.int64, size=(-1, 15)),
            'y': float
            # TODO: hardcode or use -1?
            # 'x': dict(dtype=torch.int64, size=(64, 13)),
            # 'edge_index': dict(dtype=torch.int64, size=(2, 1856)),
            # 'edge_attr': dict(dtype=torch.int64, size=(1856, 15)),
        }

        super().__init__(root, transform, pre_transform, schema=schema)

        # split (self.raw_paths[1]) contains a dictionary with keys train,val,test
        # and the data indices which correspond to these keys
        split_idx = fs.torch_load(self.raw_paths[1])
        self._indices = split_idx[self.split].tolist()

    @property
    def raw_file_names(self) -> List[str]:
        return ['first_10k_evaluations.csv', 'ranges.pt']

    def download(self) -> None:
        pass

    def process(self) -> None:
        df = pd.read_csv(self.raw_paths[0])
        data_list: List[Data] = []

        iterator = enumerate(zip(df['fen'], df['evaluation']))

        for i, (fen, evaluation) in tqdm(iterator, total=len(df)):
            data = self.from_fens(fen, evaluation, self.edges)
            data_list.append(data)
            if i + 1 == len(df) or (i + 1) % 1000 == 0:  # Write batch-wise:
                self.extend(data_list)
                data_list = []

    def serialize(self, data: BaseData) -> Dict[str, Any]:
        assert isinstance(data, Data)
        return dict(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            y=data.y
        )

    def deserialize(self, data: Dict[str, Any]) -> Data:
        return Data.from_dict(data)


def create_split_dict(dataset):
    dataset = pd.read_csv(dataset)

    total_rows = len(dataset)
    print(total_rows)

    train_ratio = 0.7
    val_ratio = 0.15

    # Compute the lengths
    train_len = int(total_rows * train_ratio)
    val_len = int(total_rows * val_ratio)

    ranges = {
        'train': torch.arange(0, train_len),
        'val': torch.arange(train_len, train_len + val_len),
        'test': torch.arange(train_len + val_len, total_rows)
    }

    # Save the dictionary to a file
    torch.save(ranges, 'ranges.pt')
