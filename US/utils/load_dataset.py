import torch
import numpy as np
import os.path as osp
from typing import Union, Tuple, Callable, Optional
from torch_geometric.data import Data, DataLoader, InMemoryDataset, download_url
          
class TRAVELDataset(InMemoryDataset):
    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return self.root  # use base path directly

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')  # central processed folder

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return f'{self.name}_data.pt'
    
    def process(self):
        data = read_npz(osp.join(self.raw_dir, self.raw_file_names))
        data = data if self.pre_transform is None else self.pre_transform(data)
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])
 
    def __repr__(self) -> str:
        return f'{self.name.capitalize()}Full()'

def read_npz(path):
    with np.load(path, allow_pickle=True) as f:
        return parse_npz(f)


def parse_npz(f):
    crash_time = f['crash_time']
    x = torch.from_numpy(f['x']).to(torch.float)
    coords = torch.from_numpy(f['coordinates']).to(torch.float)
    edge_attr = torch.from_numpy(f['edge_attr']).to(torch.float)
    cnt_labels = torch.from_numpy(f['cnt_labels']).to(torch.long)
    occur_labels = torch.from_numpy(f['occur_labels']).to(torch.long)
    edge_attr_dir = torch.from_numpy(f['edge_attr_dir']).to(torch.float)
    edge_attr_ang = torch.from_numpy(f['edge_attr_ang']).to(torch.float)
    severity_labels = torch.from_numpy(f['severity_8labels']).to(torch.long)
    edge_index = torch.from_numpy(f['edge_index']).to(torch.long).t().contiguous()
    return Data(x=x, y=occur_labels, severity_labels=severity_labels, edge_index=edge_index, 
                edge_attr=edge_attr, edge_attr_dir=edge_attr_dir, edge_attr_ang=edge_attr_ang, 
                coords=coords, cnt_labels=cnt_labels, crash_time=crash_time)


dataset = TRAVELDataset('../data/TAP-city', 'Dallas_TX')


data = dataset[0]
print(f'Number of graphs: {len(dataset)}')
print(f'Number of node features: {dataset.num_features}')
print(f'Number of edge features: {dataset.num_edge_features}')
print(f'Number of classes: {dataset.num_classes}')
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.has_isolated_nodes()}')
print(f'Contains self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')


print(data.edge_attr) 