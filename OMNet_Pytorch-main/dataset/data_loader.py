import os
from os.path import isfile, join
import pickle
import random
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from open3d import io

from dataset.transformations import fetch_transform

_logger = logging.getLogger(__name__)


def worker_init_fn(worker_id):
    rand_seed = random.randint(0, 2**32 - 1)
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)


class ModelNetNpy(Dataset):
    def __init__(
        self,
        dataset_path: str,
        dataset_mode: str,
        subset: str = "train",
        categories=None,
        transform=None,
    ):
        """ModelNet40 TS dataset."""
        self._logger = logging.getLogger(self.__class__.__name__)
        self._root = dataset_path
        self._subset = subset

        metadata_fpath = os.path.join(
            self._root, "modelnet_{}_{}.pickle".format(dataset_mode, subset)
        )
        self._logger.info(
            "Loading dataset from {} for {}".format(metadata_fpath, subset)
        )

        if not os.path.exists(os.path.join(dataset_path)):
            assert FileNotFoundError("Not found dataset_path: {}".format(dataset_path))

        with open(os.path.join(dataset_path, "shape_names.txt")) as fid:
            self._classes = [label.strip() for label in fid]
            self._category2idx = {e[1]: e[0] for e in enumerate(self._classes)}
            self._idx2category = self._classes

        if categories is not None:
            categories_idx = [self._category2idx[c] for c in categories]
            self._logger.info("Categories used: {}.".format(categories_idx))
            self._classes = categories
        else:
            categories_idx = None
            self._logger.info("Using all categories.")

        self._data = self._read_pickle_files(
            os.path.join(
                dataset_path, "modelnet_{}_{}.pickle".format(dataset_mode, subset)
            ),
            categories_idx,
        )

        self._transform = transform
        self._logger.info("Loaded {} {} instances.".format(len(self._data), subset))

    @property
    def classes(self):
        return self._classes

    @staticmethod
    def _read_pickle_files(fnames, categories):

        all_data_dict = []
        with open(fnames, "rb") as f:
            data = pickle.load(f)

        for category in categories:
            all_data_dict.extend(data[category])

        return all_data_dict

    def to_category(self, i):
        return self._idx2category[i]

    def __getitem__(self, item):

        data_path = self._data[item]

        # load and process dataset
        points = np.load(data_path)
        idx = np.array(
            int(os.path.splitext(os.path.basename(data_path))[0].split("_")[1])
        )
        label = np.array(
            int(os.path.splitext(os.path.basename(data_path))[0].split("_")[3])
        )
        sample = {"points": points, "label": label, "idx": idx}

        if self._transform:
            sample = self._transform(sample)
        return sample

    def __len__(self):
        return len(self._data)


class VirtualObjects(Dataset):
    def __init__(
        self,
        dataset_path: str,
        dataset_mode: str,
        subset: str = "train",
        categories=None,
        transform=None,
    ):
        """Virtual Objects created by Gregor in dataset form."""
        dataset_path = "/home/cracee/Documents/Optonic_Project/OptonicsObjectReconstruction/registration/data/14_ramp_order"

        self._logger = logging.getLogger(self.__class__.__name__)
        self._root = dataset_path
        self._subset = subset

        metadata_fpath = os.path.join(
            self._root, "modelnet_{}_{}.pickle".format(dataset_mode, subset)
        )
        self._logger.info(
            "Loading dataset from {} for {}".format(metadata_fpath, subset)
        )
        if not os.path.exists(os.path.join(dataset_path)):
            assert FileNotFoundError("Not found dataset_path: {}".format(dataset_path))

        self._data = self._read_folder(dataset_path)
        self._classes = None
        self._idx2category = None
        self.eval_type = ["test"]

        self._transform = transform
        self._logger.info("Loaded {} {} instances.".format(len(self._data), subset))

    @property
    def classes(self):
        return self._classes

    @staticmethod
    def _read_folder(folder_name):
        onlyfiles = [f for f in os.listdir(folder_name) if isfile(join(folder_name, f))]

        return onlyfiles

    def to_category(self, i):
        return self._idx2category[i]

    def __getitem__(self, item):

        # load the first item
        data_path = self._data[item]
        meta_path = self._root + "/" + data_path
        point_cloud = io.read_point_cloud(meta_path)
        numpy_points_1 = np.asarray(point_cloud.points)
        numpy_points_1 = (numpy_points_1 - np.min(numpy_points_1)) / (np.max(numpy_points_1) - np.min(numpy_points_1))
        numpy_points_1 = (numpy_points_1 * 2) - 1

        # load the second item
        x = (item + 1) % len(self._data)
        print(x)
        data_path = self._data[x]
        meta_path = self._root + "/" + data_path
        point_cloud = io.read_point_cloud(meta_path)
        numpy_points_2 = np.asarray(point_cloud.points)
        numpy_points_2 = (numpy_points_2 - np.min(numpy_points_2)) / (np.max(numpy_points_2) - np.min(numpy_points_2))
        numpy_points_2 = (numpy_points_2 * 2) - 1

        idx = int(data_path[-5:-4])
        sample = {"points_1": numpy_points_1, "points_2": numpy_points_2, "label": np.array([0]), "idx": idx}

        if self._transform:
            sample = self._transform(sample)
        return sample

    def __len__(self):
        return len(self._data)


def fetch_dataloader(params):
    _logger.info(
        "Dataset type: {}, transform type: {}".format(
            params.dataset_type, params.transform_type
        )
    )
    train_transforms, test_transforms = fetch_transform(params)
    if params.dataset_type == "modelnet_os":
        dataset_path = "./dataset/data/modelnet_os"
        train_categories = [
            line.rstrip("\n")
            for line in open("./dataset/data/modelnet40_half1_rm_rotate.txt")
        ]
        val_categories = [
            line.rstrip("\n")
            for line in open("./dataset/data/modelnet40_half1_rm_rotate.txt")
        ]
        test_categories = [
            line.rstrip("\n")
            for line in open("./dataset/data/modelnet40_half2_rm_rotate.txt")
        ]
        train_categories.sort()
        val_categories.sort()
        test_categories.sort()
        train_ds = ModelNetNpy(
            dataset_path,
            dataset_mode="os",
            subset="train",
            categories=train_categories,
            transform=train_transforms,
        )
        val_ds = ModelNetNpy(
            dataset_path,
            dataset_mode="os",
            subset="val",
            categories=val_categories,
            transform=test_transforms,
        )
        test_ds = ModelNetNpy(
            dataset_path,
            dataset_mode="os",
            subset="test",
            categories=test_categories,
            transform=test_transforms,
        )

    elif params.dataset_type == "modelnet_ts":
        dataset_path = "./dataset/dataset/modelnet_ts"
        train_categories = [
            line.rstrip("\n")
            for line in open("./dataset/dataset/modelnet40_half1_rm_rotate.txt")
        ]
        val_categories = [
            line.rstrip("\n")
            for line in open("./dataset/dataset/modelnet40_half1_rm_rotate.txt")
        ]
        test_categories = [
            line.rstrip("\n")
            for line in open("./dataset/dataset/modelnet40_half2_rm_rotate.txt")
        ]
        train_categories.sort()
        val_categories.sort()
        test_categories.sort()
        train_ds = ModelNetNpy(
            dataset_path,
            dataset_mode="ts",
            subset="train",
            categories=train_categories,
            transform=train_transforms,
        )
        val_ds = ModelNetNpy(
            dataset_path,
            dataset_mode="ts",
            subset="val",
            categories=val_categories,
            transform=test_transforms,
        )
        test_ds = ModelNetNpy(
            dataset_path,
            dataset_mode="ts",
            subset="test",
            categories=test_categories,
            transform=test_transforms,
        )

    elif params.dataset_type == "rampshere_big":
        dataset_path = "./dataset/data/modelnet_os"
        test_categories = [
            line.rstrip("\n")
            for line in open("./dataset/data/modelnet40_half2_rm_rotate.txt")
        ]
        test_categories.sort()
        test_ds = VirtualObjects(
            dataset_path,
            dataset_mode="os",
            subset="test",
            categories=test_categories,
            transform=test_transforms,
        )
    else:
        raise NotImplementedError

    dataloaders = {}
    params.prefetch_factor = 5
    # add default train dataset loader
    train_dl = DataLoader(
        test_ds,
        batch_size=params.train_batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        pin_memory=params.cuda,
        drop_last=True,
        prefetch_factor=params.prefetch_factor,
        worker_init_fn=worker_init_fn,
    )
    dataloaders["train"] = train_dl

    # choose val or test dataset loader for evaluate
    dl = DataLoader(
        test_ds,
        batch_size=params.eval_batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=params.cuda,
        prefetch_factor=params.prefetch_factor,
    )

    dataloaders["test"] = dl
    dataloaders["val"] = dl

    return dataloaders
