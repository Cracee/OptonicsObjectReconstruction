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

from visualize.visualizer import generate_pointcloud

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


class RealMeasuredObjects(Dataset):
    def __init__(
        self,
        dataset_path: str,
        dataset_mode: str,
        subset: str = "train",
        categories=None,
        transform=None,
    ):
        """Virtual Objects created by Gregor in dataset form."""
        dataset_path = "/home/cracee/Documents/Optonic_Project/OptonicsObjectReconstruction/registration/data/7_cylin_order"

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

    def _read_file(self, item):
        data_path = self._data[item]
        meta_path = self._root + "/" + data_path
        point_cloud = io.read_point_cloud(meta_path)
        return np.asarray(point_cloud.points)

    def to_category(self, i):
        return self._idx2category[i]

    def other_normalise(self, points_a, points_b):
        centroid = np.mean(points_a, axis=0)
        points_a -= centroid
        furthest_distance_a = np.max(np.sqrt(np.sum(abs(points_a) ** 2, axis=-1)))

        centroid = np.mean(points_b, axis=0)
        points_b -= centroid
        furthest_distance_b = np.max(np.sqrt(np.sum(abs(points_b) ** 2, axis=-1)))

        if furthest_distance_a > furthest_distance_b:
            furthest_distance = furthest_distance_a
        else:
            furthest_distance = furthest_distance_b
        points_a /= furthest_distance
        points_b /= furthest_distance

        return points_a, points_b

    @staticmethod
    def normalise(numpy_points_1, numpy_points_2):
        min_p1 = np.min(numpy_points_1)
        min_p2 = np.min(numpy_points_2)

        if min_p1 < min_p2:
            shift = min_p2 - min_p1
            numpy_points_1 = numpy_points_1 + shift
        elif min_p2 < min_p1:
            shift = min_p1 - min_p2
            numpy_points_2 = numpy_points_2 + shift

        if np.max(numpy_points_1) - np.min(numpy_points_1) >= np.max(numpy_points_2) - np.min(numpy_points_2):
            maxi_king = np.max(numpy_points_1)
            mini_king = np.min(numpy_points_1)
            divider = maxi_king - mini_king
        else:
            maxi_king = np.max(numpy_points_2)
            mini_king = np.min(numpy_points_2)
            divider = maxi_king - mini_king

        numpy_points_1 = (numpy_points_1 - mini_king) / divider
        numpy_points_1 = (numpy_points_1 * 2) - 1

        numpy_points_2 = (numpy_points_2 - mini_king) / divider
        numpy_points_2 = (numpy_points_2 * 2) - 1

        if np.max(numpy_points_2) > np.max(numpy_points_1):
            shift = np.max(numpy_points_2) - np.max(numpy_points_1) / 2
            numpy_points_1 = numpy_points_1 + shift
        elif np.max(numpy_points_1) > np.max(numpy_points_2):
            shift = np.max(numpy_points_1) - np.max(numpy_points_2) / 2
            numpy_points_2 = numpy_points_2 + shift

        return numpy_points_1, numpy_points_2

    def __getitem__(self, item):

        # load the first item

        numpy_points_1 = self._read_file(item)

        # load the second item
        x = (item + 1) % len(self._data)
        numpy_points_2 = self._read_file(x)

        # numpy_points_1, numpy_points_2 = self.normalise(numpy_points_1, numpy_points_2)
        numpy_points_1, numpy_points_2 = self.other_normalise(numpy_points_1, numpy_points_2)

        idx = int(self._data[item][-5:-4])
        sample = {"points_1": numpy_points_1, "points_2": numpy_points_2, "label": np.array([0]), "idx": idx}

        if self._transform:
            sample = self._transform(sample)
        return sample

    def __len__(self):
        return len(self._data)


class SyntheticObjects(Dataset):
    def __init__(
        self,
        subset="test",
        transform=None,
        number_of_points=2000,
    ):
        """Virtual Objects created by Gregor in dataset form."""
        # path to the 3d Object
        # dataset_path = "/home/cracee/Documents/Optonic_Project/OptonicsObjectReconstruction/registration/data/7_cylin_order"
        dataset_path = "/home/cracee/Documents/Optonic_Project/OptonicsObjectReconstruction/data/Ramp_sphere_upscale.stl"

        self._logger = logging.getLogger(self.__class__.__name__)
        self._root = dataset_path
        self._subset = subset
        self.number_of_points = number_of_points

        if not os.path.exists(os.path.join(dataset_path)):
            assert FileNotFoundError("Not found dataset_path: {}".format(dataset_path))

        self._data = dataset_path
        self._classes = None
        self._idx2category = None
        self.eval_type = ["test"]

        self._transform = transform
        self._logger.info("Loaded {} {} instances.".format(len(self._data), subset))

    def __getitem__(self, item):

        data_path = self._root

        # load the first item
        numpy_pcd = generate_pointcloud(data_path, self.number_of_points)

        idx = item
        sample = {"points": numpy_pcd, "label": np.array([0]), "idx": idx}

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
        test_ds = RealMeasuredObjects(
            dataset_path,
            dataset_mode="os",
            subset="test",
            categories=test_categories,
            transform=test_transforms,
        )

    elif params.dataset_type == "rampshere_synthetic":
        test_ds = SyntheticObjects(
            subset="test",
            transform=test_transforms,
            number_of_points=params.num_points
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
