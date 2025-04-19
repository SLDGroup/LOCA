from utils.split_utils import get_datasets

num_users = 50
num_client_imgs = 500
proxy = 0.05

for seed in [42, 182, 342]:
    # Saves the NYUv2 test dataset
    get_datasets(dataset_name="nyuv2", num_users=num_users, seed=seed, proxy=proxy,
                 num_rooms=3, num_client_imgs=num_client_imgs, iid=True,
                 separate_rooms=True, generate_nyu_test=True)
    # Save the IID dataset + global text set and cloud data for FedCL
    get_datasets(dataset_name="nyuv2", num_users=num_users, seed=seed, proxy=proxy,
                 num_rooms=3, num_client_imgs=num_client_imgs, iid=True,
                 separate_rooms=True, generate_nyu_test=False)
    # Save the Non-IID datasets for 3 rooms, 4 rooms and 3,4, or 5 rooms
    for num_rooms in [3,4]:
        get_datasets(dataset_name="nyuv2", num_users=num_users, seed=seed, proxy=proxy,
                     num_rooms=num_rooms, iid=False, separate_rooms=True, generate_nyu_test=False)
    get_datasets(dataset_name="nyuv2", num_users=num_users, seed=seed, proxy=proxy,
                 num_rooms=[3, 4, 5], iid=False, separate_rooms=True, generate_nyu_test=False)

