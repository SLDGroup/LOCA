import copy
import re
import os
import csv
import random
import torch
import pandas as pd
import numpy as np
import _pickle as cpickle
from PIL import Image
from io import BytesIO
from zipfile import ZipFile
from itertools import permutations
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import (rotate, affine, get_dimensions, adjust_brightness, adjust_saturation,
                                               adjust_contrast, adjust_sharpness, posterize, solarize, autocontrast,
                                               equalize)
from typing import Optional, List, Tuple, Dict
from torch import Tensor
_check_pil = lambda x: isinstance(x, Image.Image)
import h5py


def load_hdf5_as_dataset(hdf5_file):
    """
    Load the images and depth labels from the HDF5 file and combine them into a dataset.

    Args:
        hdf5_file (str): Path to the HDF5 file.

    Returns:
        list: List of tuples, where each tuple contains (image_sample, depth_label).
    """
    dataset = []
    with h5py.File(hdf5_file, 'r') as hf:
        # Assuming that images and depth_labels are stored in separate datasets
        images = hf['images'][:]
        depth_labels = hf['depths'][:]

        # Combine images and depth_labels into a list of tuples
        for img, depth in zip(images, depth_labels):
            dataset.append((img, depth))

    return dataset


def load_hdf5_img_depth(filename, separate_rooms=False):
    """
    Load images and depth data from an HDF5 file.

    Args:
        filename (str): Path to the HDF5 file to load.
        noniid (bool): If True, loads images and depths as separate rooms for each client.

    Returns:
        tuple: (images, depths), where images and depths are lists or arrays depending on noniid flag.
    """
    with h5py.File(filename, 'r') as hf:
        if separate_rooms:
            # For non-IID data, load each room's data separately
            images = []
            depths = []
            room_keys = sorted([key for key in hf.keys() if 'room' in key and 'images' in key])

            for room_image_key in room_keys:
                room_idx = room_image_key.split('_')[1]  # Extract room index from the key name
                room_depth_key = f'room_{room_idx}_depths'

                room_images = hf[room_image_key][:]
                room_depths = hf[room_depth_key][:]

                images.append(room_images)
                depths.append(room_depths)
        else:
            # For IID data, load all images and depths together
            images = hf['images'][:]
            depths = hf['depths'][:]

    return images, depths


def ndarray_to_pil(image):
    """Convert a NumPy array to a PIL Image without capping the depth values."""
    # Check if the image is 2D or 3D
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert RGB-like images (3 channels) to 8-bit RGB
        return Image.fromarray(image.astype('uint8'), 'RGB')
    elif image.dtype == 'uint16':
        # Convert 16-bit grayscale images directly using 'I;16' mode
        return Image.fromarray(image, mode='I;16')
    else:
        # Default handling for other grayscale images
        return Image.fromarray(image.astype('uint8'))


class CustomNYU(Dataset):
    """
    Custom NYUv2 Dataset that applies specified transformations to images.
    """

    def __init__(self, dataset=None, imgs=None, labels=None, test=False, rotate=False, transform=None):
        """
        Initializes the dataset.

        Parameters:
            dataset (list): Preloaded dataset.
            imgs (list): List of image data.
            labels (list): List of labels.
            split (str): 'train' or 'test'.
            test (bool): Flag for test mode.
            rotate (bool): Whether to apply rotation.
            weak_strong (bool): Whether to apply weak/strong augmentations.
            transform (callable): Transformation to apply.
            seed (int): Seed for deterministic behavior.
        """
        if dataset is not None:
            self.data = [item[0] for item in dataset]
            self.targets = [item[1] for item in dataset]
        else:
            self.data = [item for item in imgs]
            self.targets = [item for item in labels]
        self.test = test
        self.transform = transform  # Assign the passed transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pre_raw_image, pre_raw_depth = self.data[idx], self.targets[idx]
        pre_raw_image = ndarray_to_pil(pre_raw_image)
        pre_raw_depth = ndarray_to_pil(pre_raw_depth)

        sample = (pre_raw_image, pre_raw_depth, pre_raw_image, pre_raw_depth)
        if self.transform:
            sample = self.transform(sample)
        raw_image, raw_depth, img, depth = sample

        if self.test:
            sample = (img, depth)
        else:
            sample = (raw_image, raw_depth, img, depth)

        return sample


def load_data(data_iid=True, dev_idx=-1, dataset_name="nyuv2", seed=42, test_global=False,
              batch_size=20, train_type=None, train_split=0.8, num_rooms="3", nyuv2_test=False):
    """
    Obtains DataLoader for each level of the framework's hierarchy

    Parameters:
    data_iid (str): IID or NIID, used in path selection
    dev_idx (int): Desired device index
    dataset_name (str): Only supports nyuv2
    seed (int): Random seed
    test_global (bool): Obtains the test set for cloud
    batch_size (int): Batch size
    labeled (int): Number of labeled images used in path selection
    """

    def worker_init_fn(worker_id):
        # Ensure each worker has a unique but reproducible seed
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)
    assert train_split == 0.8, f"Train split is {train_split}"

    # base_dir = f"dataset/{dataset_name}/{seed}_{data_type}"
    base_dir = f"dataset/{dataset_name}/seed{seed}"
    suffix = f"iid_{num_rooms}rooms" if data_iid else f"niid_{num_rooms}rooms"

    if test_global:
        # with open(f"{base_dir}/{suffix}/global_test.pkl", 'rb') as f:
        #     dataset = cpickle.load(f)
        #     data_loader = DataLoader(CustomNYU(dataset=dataset, test=True),
        #                              batch_size=128, num_workers=2, worker_init_fn=worker_init_fn, pin_memory=True, shuffle=False)
        #     return data_loader
        # Load the dataset from the HDF5 file
        if nyuv2_test:
            transform = get_transforms(split="test", seed=seed)
            hdf5_file = f"{base_dir}/nyuv2_test.h5"
        else:
            transform = get_transforms(split="val", seed=seed)
            hdf5_file = f"{base_dir}/global_test.h5"
        dataset = load_hdf5_as_dataset(hdf5_file)
        # Create an instance of CustomNYU using the loaded dataset
        custom_nyu_dataset = CustomNYU(dataset=dataset, test=True, transform=transform)

        # Create a DataLoader for the dataset
        data_loader = DataLoader(custom_nyu_dataset, batch_size=128, num_workers=2, worker_init_fn=worker_init_fn,
                                 pin_memory=True, shuffle=False)
        return data_loader
    elif train_type == "mde_fedcl":
        print(f"[+] Loading FedCL proxy dataset...")

        hdf5_file = f"{base_dir}/cloud_data.h5"
        transform = get_transforms(split="train", seed=seed)
        # Load images and depths from the HDF5 file
        images, depths = load_hdf5_img_depth(hdf5_file, separate_rooms=False)

        # Create a CustomNYU dataset instance
        custom_nyu_dataset = CustomNYU(dataset=None, imgs=images, labels=depths, transform=transform, test=False)

        # Create and return a DataLoader for the dataset
        data_loader = DataLoader(custom_nyu_dataset, batch_size=batch_size, num_workers=2,
                                 worker_init_fn=worker_init_fn, pin_memory=True, shuffle=True)
        return data_loader

    else:
        train_transform = get_transforms(split="train", seed=seed)
        val_transform = get_transforms(split="val", seed=seed)
        hdf5_file = f"{base_dir}/{suffix}/imgs_labels_dev{dev_idx}.h5"
        images, depths = load_hdf5_img_depth(hdf5_file, separate_rooms=True)

        train_loaders = []
        val_loaders = []
        for room_idx  in range(len(images)):
            # For each room, treat images[room_idx] and depths[room_idx] as the room's data
            room_images = images[room_idx]
            room_depths = depths[room_idx]
            # Full dataset instance
            full_dataset = CustomNYU(dataset=None, imgs=room_images, labels=room_depths, transform=train_transform)
            # Calculate split index
            total_size = len(full_dataset)
            split_idx = int(train_split * total_size)

            # Define train and validation indices
            train_indices = list(range(0, split_idx))
            val_indices = list(range(split_idx, total_size))

            # Create Subsets for training
            train_subset = Subset(full_dataset, train_indices)

            # Create a separate dataset instance for validation with val_transform
            val_dataset = CustomNYU(dataset=None, imgs=room_images, labels=room_depths, test=True, transform=val_transform)
            val_subset = Subset(val_dataset, val_indices)

            # Create DataLoaders
            train_loader = DataLoader(train_subset, batch_size=batch_size, num_workers=2, worker_init_fn=worker_init_fn,
                                      pin_memory=True, shuffle=data_iid)
            val_loader = DataLoader(val_subset, batch_size=batch_size, num_workers=2, worker_init_fn=worker_init_fn,
                                    pin_memory=True, shuffle=False)

            train_loaders.append(train_loader)
            val_loaders.append(val_loader)
        return train_loaders, val_loaders


def extract_room_and_number(path: str):
    """
    Helper function for sorting paths in NYUv2
    """

    match = re.match(r'data/nyu2_train/([^/]+)/(\d+)\.jpg', path) or re.match(r'data/nyu2_train/([^/]+)/(\d+)\.png', path)
    if match:
        room, number = match.groups()
        return room, int(number)
    else:
        return '', 0


def get_nyuv2(zipfile: str, split: str):
    """
    Obtain NYUv2 dataset with sorted folders (continuous data streams)

    Parameters:
    zipfile (String): Path to NYUv2 Zip File
    split (String): Training or testing split

    Returns:
    dict: A dictionary that maps room indices to a dict of folders that
    belong to the room category such as room 6 under category bathroom
    dict: A dictionary that maps room category names to encoded indices

    or

    Returns:
    list: A list with the values given in the NYUv2 test dataset
    int: -1
    """

    assert split == 'train' or split == 'test', \
        f"Split must be either \'train\' or \'test\', gave {split} instead"

    print("[+] Grabbing file bytecode from zip...")
    input_zip = ZipFile(zipfile)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}

    print(f"[+] Creating {split} split...")
    if split == 'train':
        rooms = dict()
        room_keys = dict()
        room_folders_keys = dict()

        # Sorting rooms by path
        df = pd.read_csv('nyu_data/data/nyu2_train.csv', header=None, names=['jpg', 'png'])
        df['Room'], df['Number'] = zip(*df['jpg'].apply(extract_room_and_number))

        sorted_df = df.sort_values(by=['Room', 'Number']).drop(columns=['Room', 'Number'])

        sorted_df.to_csv('sorted_nyu2_train.csv', index=False, header=False)

        with open('sorted_nyu2_train.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) > 0:
                    # Opening RGB and Depth as PIL
                    if len(row) == 1:
                        print(row,"    #    ")

                    image = os.path.join('nyu_data', row[0])
                    depth = os.path.join('nyu_data', row[1])

                    image = Image.open(BytesIO(data[image]))
                    depth = Image.open(BytesIO(data[depth]))

                    # Splitting path into major/minor classification
                    room_folder = row[0].split('/')[2]  # basement_0001a_out | conference_room_0001_out
                    room = room_folder.split('_')  # basement,0001a,out |conference,room,0001,out
                    room = '_'.join(room[:-2])  # basement | conference_room

                    # Encoding major rooms into indices
                    # Used only in the sample
                    if room not in room_keys:
                        room_idx = len(rooms.keys())  # unique number of rooms
                        room_keys[room] = room_idx
                    else:
                        room_idx = room_keys[room]

                    # Obtaining the major room at the index
                    folder = rooms.get(room_idx, dict())

                    # Encoding minor rooms into indices
                    if room_folder not in room_folders_keys:
                        room_folder_idx = len(folder.keys())
                        room_folders_keys[room_folder] = room_folder_idx
                    else:
                        room_folder_idx = room_folders_keys[room_folder]

                    # Obtaining the minor room at the index
                    dataset = folder.get(room_folder_idx, list())

                    # Sample comprised of PIL images and room classification
                    sample = ((image, depth), room_idx)

                    dataset.append(sample)

                    # Updating minor/major rooms
                    folder[room_folder_idx] = dataset
                    rooms[room_idx] = folder

        print(f"[+] NYUv2 {split} split completed!")
        return rooms, room_keys
    elif split == 'test':
        # If it is a test split, return unsorted and given NYUv2 test set
        test_dataset = list()

        for row in (data[f'nyu_data/data/nyu2_test.csv']).decode("utf-8").split('\n'):
            if len(row) > 0:
                nyu2_test = list(row.split(','))

                # Opening RGB and Depth as PIL
                image = os.path.join('nyu_data', nyu2_test[0])
                depth = os.path.join('nyu_data', nyu2_test[1])

                image = Image.open(BytesIO(data[image]))
                depth = Image.open(BytesIO(data[depth]))

                sample = ((image, depth), -1)
                test_dataset.append(sample)

        return test_dataset, -1
    return None


class RandomHorizontalFlip(object):
    def __init__(self, probability, seed=42):
        """
        Initializes the RandomHorizontalFlip transformation.

        Parameters:
            probability (float): Probability of flipping the image.
            seed (int): Fixed seed for reproducibility.
        """
        self.probability = probability
        assert 0.0 <= self.probability <= 1.0, "Probability must be between 0 and 1"
        self.rng = random.Random(seed)  # Initialize a local RNG with a fixed seed

    def __call__(self, sample):
        """
        Applies the horizontal flip transformation to the sample.

        Parameters:
            sample (tuple): A tuple containing (raw_img, raw_depth, img, depth).

        Returns:
            tuple: Transformed sample with potential horizontal flips.
        """
        raw_img, raw_depth, img, depth = sample

        if not _check_pil(img) or not _check_pil(depth):
            raise TypeError("Expected PIL type. Got {}".format(type(img)))

        if self.rng.random() < self.probability:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # raw_img = raw_img.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
            # raw_depth = raw_depth.transpose(Image.FLIP_LEFT_RIGHT)

        return (raw_img, raw_depth, img, depth)


class RandomChannelSwap(object):
    def __init__(self, probability, seed=42):
        self.probability = probability
        self.indices = list(permutations(range(3), 3))
        self.rng = random.Random(seed)  # Fixed seed for determinism

    def __call__(self, sample):
        raw_image, raw_depth, image, depth = sample

        if not _check_pil(image) or not _check_pil(depth):
            raise TypeError("Expected PIL type.")

        if self.rng.random() < self.probability:
            image_np = np.asarray(image)
            perm = self.rng.choice(self.indices)
            image_np = image_np[..., list(perm)]
            image = Image.fromarray(image_np)

        return (raw_image, raw_depth, image, depth)


class ToTensor(object):
    def __init__(self, test=False, maxDepth=1000.0):
        self.test = test
        self.maxDepth = maxDepth

    def __call__(self, sample):
        raw_image, raw_depth, image, depth = sample

        transformation = transforms.ToTensor()
        raw_image = np.array(raw_image).astype(np.float32) / 255.0
        image = np.array(image).astype(np.float32) / 255.0
        raw_depth = np.array(raw_depth).astype(np.float32)
        depth = np.array(depth).astype(np.float32)

        # The depth image should be in range of 0-10000.0
        if self.test:
            depth = depth / 1000.0
            raw_depth = raw_depth / 1000.0

            image, depth = transformation(image), transformation(depth)
            raw_image, raw_depth = transformation(raw_image), transformation(raw_depth)
        # The depth image should be a PIL image -> np
        else:
            depth = depth / 255.0 * 10.0
            raw_depth = raw_depth / 255.0 * 10.0

            valid_mask = depth != 0
            depth[valid_mask] = self.maxDepth / depth[valid_mask]
            raw_depth[valid_mask] = self.maxDepth / raw_depth[valid_mask]
            # depth = self.maxDepth / depth
            image, depth = transformation(image), transformation(depth)
            raw_image, raw_depth = transformation(raw_image), transformation(raw_depth)

            zero_mask = depth == 0
            depth = torch.clamp(depth, self.maxDepth/100.0, self.maxDepth)
            depth[zero_mask] = 0.0

            zero_mask = raw_depth == 0
            raw_depth = torch.clamp(raw_depth, self.maxDepth/100.0, self.maxDepth)
            raw_depth[zero_mask] = 0.0

        image = torch.clamp(image, 0.0, 1.0)
        raw_image = torch.clamp(raw_image, 0.0, 1.0)
        return (raw_image, raw_depth, image, depth)


def get_transforms(split, seed=42):
    if split == 'train':
        t = transforms.Compose([RandomHorizontalFlip(probability=0.5, seed=seed),
                                RandomChannelSwap(probability=0.25, seed=seed),
                                ToTensor(test=False, maxDepth=10.0)])
    elif split == 'val':
        t = transforms.Compose([ToTensor(test=False, maxDepth=10.0)])
    elif split == 'test':
        t = transforms.Compose([ToTensor(test=True, maxDepth=10.0)])

    return t
