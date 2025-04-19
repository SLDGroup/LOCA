import os
import random
import numpy as np
from utils.dataset_utils import get_nyuv2
from collections import Counter
from utils.general_utils import seed_everything
from tqdm import tqdm
import h5py
import _pickle as cpickle


def save_pickle(file_path, data):
    with open(file_path, 'wb') as f:
        cpickle.dump(data, f)


def save_hdf5_dataset(filename, cloud_dataset):
    """
    Save the cloud_dataset to an HDF5 file.

    Args:
        filename (str): The name of the file to save the dataset.
        cloud_dataset (list): List of tuples, where each tuple contains (image_sample, depth_label).
    """
    # Convert the cloud dataset to numpy arrays for storage in HDF5
    images = np.array([img for (img, _), _ in cloud_dataset])
    depths = np.array([depth for (_, depth), _ in cloud_dataset])

    # Create HDF5 file and datasets
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset('images', data=images)
        hf.create_dataset('depths', data=depths)

    print(f"Dataset saved to {filename}")


def save_hdf5_img_depth(filename, images, depths, separate_rooms=False):
    """
    Save images and depth data to an HDF5 file.

    Args:
        filename (str): Name of the HDF5 file to save the data.
        images (list): List of numpy arrays for image data, can be nested lists for separate rooms.
        depths (list): List of numpy arrays for depth data, can be nested lists for separate rooms.
        separate_rooms (bool): If True, handle images and depths as separate rooms.
    """
    with h5py.File(filename, 'w') as hf:
        if separate_rooms:
            # Save images and depths for each room separately
            for room_idx, (room_images, room_depths) in enumerate(zip(images, depths)):
                # Create group for each room and save the list of all images and depths as a single dataset
                hf.create_dataset(f'room_{room_idx}_images', data=np.array(room_images))
                hf.create_dataset(f'room_{room_idx}_depths', data=np.array(room_depths))
        else:
            # Save all images and depths together as single datasets
            hf.create_dataset('images', data=np.array(images))
            hf.create_dataset('depths', data=np.array(depths))

    print(f"Dataset saved to {filename}")


def initialize_directories(base_dir, suffix):
    dir_path = os.path.join(base_dir, suffix)
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def process_cloud_dataset(cloud_fedcl_subset, base_dir, proxy):
    total_cloud_subset_data = []
    total_cloud_subset_labels = []
    total_cloud_subset_room_idx = []

    for key, values in cloud_fedcl_subset.items():
        cloud_subset_data = [img for (img, _), _ in values]
        cloud_subset_labels = [depth for (_, depth), _ in values]
        cloud_subset_room_idx = [room_idx for (_, _), room_idx in values]
        total_cloud_subset_data.extend(cloud_subset_data)
        total_cloud_subset_labels.extend(cloud_subset_labels)
        total_cloud_subset_room_idx.extend(cloud_subset_room_idx)

    combined = list(zip(total_cloud_subset_data, total_cloud_subset_labels, total_cloud_subset_room_idx))
    sample_size = int(len(combined) * 0.2)  # From 5% data use 1% for FedCL
    if sample_size < len(combined):
        samples = random.sample(combined, sample_size)
    else:
        samples = combined
    total_cloud_subset_data, total_cloud_subset_labels, total_cloud_subset_room_idx = zip(*samples)
    # Save the data and labels to HDF5 file instead of pickle
    save_hdf5_img_depth(os.path.join(base_dir, "cloud_data.h5"), images=total_cloud_subset_data, depths=total_cloud_subset_labels, separate_rooms=False)

    # Print class frequency
    class_freq = Counter(total_cloud_subset_room_idx)
    ordered_class_freq = dict(sorted(class_freq.items()))
    print(f"Cloud {proxy * 100 * 0.2}%:\n\t\tTrain: {ordered_class_freq}\n\t\tTotal: {sum(ordered_class_freq.values())}")


def save_client_datasets(total_client_dataset, dir_path, separate_rooms=False):
    """Save client datasets using HDF5 format."""
    client_dist = {}
    for client_num, client_data_list in enumerate(total_client_dataset):
        total_client_samples = []
        total_client_labels = []
        total_client_room_idx = []

        # Combine data from all simulated rooms for this client
        for client_data in client_data_list:
            client_samples = [img for (img, _), _ in client_data]
            client_labels = [depth for (_, depth), _ in client_data]
            client_room_idx = [room_idx for (_, _), room_idx in client_data]

            total_client_samples.append(client_samples)
            total_client_labels.append(client_labels)
            total_client_room_idx.append(client_room_idx)

        client_nk = sum(len(samples) for samples in total_client_samples)
        client_dist[client_num] = client_nk

        class_freq = Counter([idx for sublist in total_client_room_idx for idx in sublist])
        ordered_class_freq = dict(sorted(class_freq.items()))

        # Print the ordered class frequency
        print(f"\tClient {client_num}:\n\t\tTrain: {ordered_class_freq}\n\t\tTotal: {sum(ordered_class_freq.values())}")

        # Save data to HDF5 file instead of pickle
        save_hdf5_img_depth(os.path.join(dir_path, f"imgs_labels_dev{client_num}.h5"), images=total_client_samples,
                            depths=total_client_labels, separate_rooms=separate_rooms)

    save_pickle(os.path.join(dir_path, "client_dist.pkl"), client_dist)


def get_datasets(dataset_name: str, num_users: int, seed: int, proxy: float, num_rooms,
                 num_client_imgs=500, iid=False, separate_rooms=True, generate_nyu_test=False):
    seed_everything(seed=seed)

    base_dir = f"dataset/{dataset_name}/seed{seed}"
    if isinstance(num_rooms, int):
        suffix = f"iid_{num_rooms}rooms" if iid else f"niid_{num_rooms}rooms"
    elif isinstance(num_rooms, list):
        if iid:
            print("IID not supported for list")
        else:
            nm = "".join(map(str, num_rooms))
            suffix = f"niid_{nm}rooms"

    dir_path = initialize_directories(base_dir, suffix)
    if generate_nyu_test:
        dataset, _ = get_nyuv2(zipfile="nyuv2.zip", split="test")
        save_hdf5_dataset(os.path.join(base_dir, "nyuv2_test.h5"), dataset)
        print(f"Created NYUv2 TEST dataset of length {len(dataset)}")
        return 0

    dataset, major_room_keys = get_nyuv2(zipfile="nyuv2.zip", split="train")

    cloud_dataset = []
    cloud_fedcl_subset = {}

    # FedCL proxy% created then cloud dataset receives rest of images for each folder.
    for major_room in major_room_keys:
        major_room_idx = major_room_keys[major_room]
        next_idx = len(dataset[major_room_idx]) - 1

        shared_dataset = dataset[major_room_idx][next_idx]
        if proxy > 0.0:
            proxy_num = int(len(shared_dataset) * proxy)
            cloud_fedcl_subset[major_room_idx] = shared_dataset[:proxy_num]
        else:
            proxy_num = 0
        cloud_dataset.extend(shared_dataset[proxy_num:])

    # Removing any folders that were used in creating cloud/edge datasets
    for i in list(cloud_fedcl_subset.keys()):
        next_idx = len(dataset[i]) - 1
        del dataset[i][next_idx]
        if len(dataset[i]) == 0:
            del dataset[i]

    if iid:
        # Save cloud dataset
        if not os.path.isfile(os.path.join(base_dir, "global_test.h5")):
            save_hdf5_dataset(os.path.join(base_dir, "global_test.h5"), cloud_dataset)
            print(f"Created cloud dataset of length {len(cloud_dataset)}")

        # Process cloud dataset
        if not os.path.isfile(os.path.join(base_dir, "cloud_data.h5")):
            # Process cloud dataset
            process_cloud_dataset(cloud_fedcl_subset, base_dir, proxy)

        # Flatten the dataset
        total_dataset = []
        for v in dataset.values():
            for samples in v.values():
                total_dataset.extend(samples)

        # Shuffle total_dataset for randomness
        random.shuffle(total_dataset)

        # Create a list of indices to track available images
        total_indices = list(range(len(total_dataset)))
        used_indices = set()

        total_client_dataset = []
        imgs_per_room = int(num_client_imgs / num_rooms)

        for idx in range(num_users):
            client_data_list = []
            for room_idx in range(num_rooms):
                # Ensure no overlap between selected samples and the used indices
                available_indices = list(set(total_indices) - used_indices)
                if len(available_indices) < imgs_per_room:
                    print(f"Warning: Not enough unique images left to satisfy {imgs_per_room} images for client {idx}, room {room_idx}")
                    break

                # Select samples for the current client and room without replacement
                sample_indices = np.random.choice(available_indices, size=imgs_per_room, replace=False)
                client_dataset = [total_dataset[i] for i in sample_indices]

                # Update used_indices to avoid repetition across clients and rooms
                used_indices.update(sample_indices)
                total_indices = list(set(total_indices) - used_indices)

                client_data_list.append(client_dataset)

            total_client_dataset.append(client_data_list)

        os.makedirs(base_dir, exist_ok=True)
        print("\nSampling configuration:")
        print(f"\tDataset: {dataset_name}")
        print(f"\tNumber of clients: {num_users}")
        print(f"\tDistribute IID: True")
        print(f"\tWriting data at this location: {base_dir}")

        # Save client datasets
        save_client_datasets(total_client_dataset, dir_path, separate_rooms=separate_rooms)
    else:
        debug_rooms = set()
        total_client_dataset = []

        for _ in tqdm(range(num_users)):
            client_data_list = []

            # If num_rooms is a list, pick a random number of rooms from it for this client
            if isinstance(num_rooms, list):
                client_num_rooms = np.random.choice(num_rooms)
            else:
                client_num_rooms = num_rooms  # If it's an int, use it directly

            # Refresh valid room types and count folders for each room type
            valid_room_types = {room_type: len(folders) for room_type, folders in dataset.items() if len(folders) > 0}

            # Check if there are enough distinct room types with sufficient folders for this client
            available_room_types_with_folders = [room_type for room_type, count in valid_room_types.items() if count >= 1]

            if len(available_room_types_with_folders) < client_num_rooms:
                print(f"Error: Not enough distinct room types left with sufficient folders to satisfy {client_num_rooms} rooms for this client.")
                exit(-1)

            # Calculate weights based on the number of folders in each room type (more folders = higher probability)
            folder_counts = np.array([valid_room_types[room_type] for room_type in available_room_types_with_folders])
            probabilities = folder_counts / folder_counts.sum()

            # Preferentially select room types based on the calculated probabilities
            client_chosen_room_types = np.random.choice(available_room_types_with_folders, size=client_num_rooms,
                                                        replace=False, p=probabilities)

            for sample_room_type in client_chosen_room_types:
                # Choose a folder from the selected room type
                sample_choice = np.random.choice(list(dataset[sample_room_type].keys()), 1, replace=False)[0]

                local_data = dataset[sample_room_type][sample_choice]
                client_data_list.append(local_data)

                # Remove the selected folder (room) from the dataset
                del dataset[sample_room_type][sample_choice]

                # If no folders are left in this room type, remove it from the dataset
                if len(dataset[sample_room_type]) == 0:
                    del dataset[sample_room_type]

                # Check for duplicates in debug_rooms
                if (sample_room_type, sample_choice) in debug_rooms:
                    print("Repeat found:")
                    print("\t\t", (sample_room_type, sample_choice))
                    exit(0)
                else:
                    debug_rooms.add((sample_room_type, sample_choice))

            total_client_dataset.append(client_data_list)

        print("Total dataset per client created successfully.")

        os.makedirs(base_dir, exist_ok=True)
        print("\nSampling configuration:")
        print(f"\tDataset: {dataset_name}")
        print(f"\tNumber of clients: {num_users}")
        print(f"\tDistribute IID: False")
        print(f"\tWriting data at this location: {base_dir}")

        # Save client datasets
        save_client_datasets(total_client_dataset, dir_path, separate_rooms=separate_rooms)

    print("[+] Datasets for each user saved successfully.")
    return "done"
