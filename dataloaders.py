import os.path as osp
import pickle
from typing import Union

import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix, vstack

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import utilities


class SparseDataset:
    """
    Custom Dataset class for scipy sparse matrix

    Found from https://discuss.pytorch.org/t/dataloader-loads-data-very-slow-on-sparse-tensor/117391/4
    """

    def __init__(
        self,
        data: Union[np.ndarray, coo_matrix, csr_matrix],
        targets: Union[np.ndarray, coo_matrix, csr_matrix],
        transform: bool = None,
    ):
        # Transform data coo_matrix to csr_matrix for indexing
        if type(data) == coo_matrix:
            self.data = data.tocsr()
        else:
            self.data = data

        # Transform targets coo_matrix to csr_matrix for indexing
        if type(targets) == coo_matrix:
            self.targets = targets.tocsr()
        else:
            self.targets = targets

        self.transform = transform  # Can be removed

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]

    def get_all_data(self):
        return self.data, self.targets


def sparse_coo_to_tensor(coo: coo_matrix):
    """
    Transform scipy coo matrix to pytorch sparse tensor
    """
    values = coo.data
    indices = (coo.row, coo.col)  # np.vstack
    shape = coo.shape

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    s = torch.Size(shape)

    return torch.sparse.FloatTensor(i, v, s).to(DEVICE)


def sparse_batch_collate(batch):
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    # batch[0] since it is returned as a one element list
    data_batch, targets_batch = batch[0]

    if type(data_batch[0]) == csr_matrix:
        data_batch = data_batch.tocoo()  # removed vstack
        data_batch = sparse_coo_to_tensor(data_batch)
    else:
        data_batch = torch.FloatTensor(data_batch).to(DEVICE)

    if type(targets_batch[0]) == csr_matrix:
        targets_batch = targets_batch.tocoo()  # removed vstack
        targets_batch = sparse_coo_to_tensor(targets_batch)
    else:
        targets_batch = torch.FloatTensor(targets_batch).to(DEVICE)
    return data_batch, targets_batch


def load_data(dataset_name, data_dir_path="./data"):
    """
    Load data from dataset name

    :param dataset_name:
    :param split_validation_data:
    :return:
    """
    path = osp.normpath(f"./{data_dir_path}/{dataset_name}")
    assert osp.exists(path), f"Dataset path {path} does not exist!"
    train_test_data = pickle.load(
        open(osp.normpath(f"{path}/{dataset_name}_train.pkl"), "rb")
    )
    valid_data_data = pickle.load(
        open(osp.normpath(f"{path}/{dataset_name}_valid.pkl"), "rb")
    )
    val_train, _ = utilities.split_train_test_proportion_from_csr_matrix(
        valid_data_data, batch_size=1000, random_seed=123, test_prop=0.2
    )
    train_test_partial_valid_data = vstack((train_test_data, val_train))
    return train_test_data, train_test_partial_valid_data, valid_data_data
