import os
import torch
import argparse
import json
from models.get_model import get_model
from utils.train_test import global_test
from utils.general_utils import get_hw_info, seed_everything
from utils.fl_utils import aggregate_avg, aggregate_cos
from utils.fedcl.fedcl_utils import ewc
import _pickle as cpickle
import time
from device_handler import DeviceHandler
import numpy as np
from os import path

class Cloud:
    def __init__(self, cloud_cfg, dev_cfg, seed):
        self.cloud_cfg = cloud_cfg
        self.dev_cfg = dev_cfg
        self.seed = seed
        os.makedirs("logs", exist_ok=True)
        seed_everything(seed=seed)

        # Initialize the RNG once in the constructor
        self.device_rng = np.random.default_rng(seed=seed)  # Initialized once

    def federated_learning(self):
        total_time_start = time.time()
        with open(self.cloud_cfg, 'r') as cfg:
            dat = json.load(cfg)

            exp = dat["experiment"]
            r = dat["run"]
            base_dir = f"logs/{exp}_{r}"
            os.makedirs(base_dir, exist_ok=True)
            log_file = f"log_{exp}_{r}.csv"
            nyuv2_log_file = f"nyuv2_log_{exp}_{r}.csv"

            files_number = dat["files_number"]

            cloud_ip = dat["cloud_ip"]
            cloud_port = dat["cloud_port"]
            cloud_cuda_name = dat["cloud_cuda_name"]
            cloud_pwd, cloud_usr, cloud_path, eval_path = get_hw_info(hw_type=files_number)
            os.makedirs(cloud_path, exist_ok=True)
            os.makedirs(eval_path, exist_ok=True)

            model_name = dat["model_name"]  # {dataset_name}_{model_name}
            train_type = dat["train_type"]
            comm_rounds = dat["comm_rounds"]
            verbose = dat["verbose"]

            num_users = dat["num_users"]
            dev_fraction = dat["dev_fraction"]
            learning_rate = dat["learning_rate"]
            batch_size = dat["batch_size"]

            N_fcl = dat["N_fcl"]
            coe = dat['coe']
            dataset_name = model_name.split('_')[0]
            kt = dat["kt"]

            num_rooms = dat["num_rooms"]
            data_iid = dat["data_iid"]

            comm_round_time = 0
            comm_round_time_lst = []
            comm_round_time_avg = 0
            
            net_glob = get_model(model_name=f"{model_name}", seed=self.seed, cuda_name=cloud_cuda_name)

        base_dir_dataset = f"dataset/{dataset_name}/seed{self.seed}"
        suffix = f"iid_{num_rooms}rooms" if data_iid else f"niid_{num_rooms}rooms"

        optimizer = torch.optim.AdamW(net_glob.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=comm_rounds, eta_min=learning_rate/100.)

        # Save global weights
        torch.save(net_glob.state_dict(), path.join(cloud_path, f"global_weights.pth"))

        # Test global model
        print(f"[+] Loaded things. Testing...")
        loss_test, metrics_test = global_test(model=net_glob, cuda_name=cloud_cuda_name, dataset_name=dataset_name,
                                              seed=self.seed, num_rooms=num_rooms, nyuv2_test=False)
        print(f"Initial Deltas: d1 {metrics_test[4].avg:.4f} | d2 {metrics_test[5].avg:.4f} | d3 {metrics_test[6].avg:.4f}; Initial Loss: {loss_test:.4f}")
        with open(path.join(f"{base_dir}", log_file), 'w') as logger:
            logger.write(f"CommRound,Abs_Rel,Sq_Rel,RMSE,RMSE_Log,d1,d2,d3,Loss\n0,{metrics_test[0].avg},{metrics_test[1].avg},{metrics_test[2].avg},{metrics_test[3].avg},{metrics_test[4].avg},{metrics_test[5].avg},{metrics_test[6].avg},{loss_test}\n")

        loss_test, metrics_test = global_test(model=net_glob, cuda_name=cloud_cuda_name, dataset_name=dataset_name,
                                              seed=self.seed, num_rooms=num_rooms, nyuv2_test=True)
        print(f"Initial NYUv2 Deltas: d1 {metrics_test[4].avg:.4f} | d2 {metrics_test[5].avg:.4f} | d3 {metrics_test[6].avg:.4f}; Initial Loss: {loss_test:.4f}")
        with open(path.join(f"{base_dir}", nyuv2_log_file), 'w') as logger:
            logger.write(f"CommRound,Abs_Rel,Sq_Rel,RMSE,RMSE_Log,d1,d2,d3,Loss\n0,{metrics_test[0].avg},{metrics_test[1].avg},{metrics_test[2].avg},{metrics_test[3].avg},{metrics_test[4].avg},{metrics_test[5].avg},{metrics_test[6].avg},{loss_test}\n")

        with open(path.join(base_dir_dataset, suffix, 'client_dist.pkl'), 'rb') as f:
            client_dist = cpickle.load(f)

        for comm_round in range(comm_rounds):
            start_time = time.time()
            remaining_time = comm_round_time_avg * (comm_rounds-(comm_round+1))
            print(f"{exp}, {r}, {comm_round}, {train_type}, {model_name}:{cloud_cuda_name}, "
                  f"lr={learning_rate} "
                  f"TIME:{int(comm_round_time // 60)}m {int(comm_round_time % 60)}s, Avg TIME:{int(comm_round_time_avg // 60)}m {int(comm_round_time_avg % 60)}s, "
                  f"Approx remaining time: {int(remaining_time // 3600)}h {int((remaining_time % 3600) // 60)}m {int(remaining_time % 60)}s")
            global_weights = torch.load(path.join(cloud_path, f"global_weights.pth"))
            net_glob.load_state_dict(global_weights)
            with open(self.dev_cfg, 'r') as f:
                dt = json.load(f)
            num_devices = dt["num_devices"]
            if verbose:
                print(f"Number of devices simulated at the same time: {num_devices}")

            # split available_devices_idx in series of <num_devices> to run in parallel
            mycnt = 0
            mydictcnt = 0
            avail_devs_idx_dict = {}
            # Obtain new number of devices to consider for this communication round

            # Function to select a random subset of devices
            available_devices = self.device_rng.choice(range(num_users), size=int(dev_fraction * num_users), replace=False)

            for myidx in available_devices:
                if mycnt == 0:
                    avail_devs_idx_dict[mydictcnt] = [myidx]
                else:
                    avail_devs_idx_dict[mydictcnt].append(myidx)
                mycnt += 1
                if mycnt == num_devices:
                    mycnt = 0
                    mydictcnt += 1

            # Only if using FedCL
            if (comm_round+1) % N_fcl == 0 and train_type == "mde_fedcl":
                if verbose:
                    print(f"[!] Creating w_d for FedCL")
                w_d = ewc(model=net_glob, train_type=train_type, cuda_name=cloud_cuda_name, dataset_name=dataset_name,
                          seed=self.seed, learning_rate=learning_rate, batch_size=16, num_rooms=num_rooms)
                torch.save(w_d, path.join(cloud_path, f"comm_w_d.pth"))
                if comm_round >= 99:
                    torch.save(w_d, path.join(eval_path, f"comm_w_d.pth"))

            # net_local = copy.deepcopy(net_glob)
            for mydict_key in avail_devs_idx_dict.keys():
                device_handler_list = []
                for idx, i in enumerate(avail_devs_idx_dict[mydict_key]):
                    dev = dt[f"dev{idx + 1}"]
                    dev_type = dev["hw_type"]
                    local_epochs = dev["local_epochs"]
                    dev_host = dev["hostname"]
                    dev_port = dev["port"]
                    dev_cuda_name = dev["cuda_name"]
                    dev_model_name = dev["model_name"]

                    dev_model_filename = f"dev_{i}.pth"
                    torch.save(net_glob.state_dict(), path.join(cloud_path, dev_model_filename))
                    dev_pwd, dev_usr, dev_path, eval_dev_path = get_hw_info(dev_type)

                    """
                    setup_message = {cloud_info},{device_info},{data}

                    where

                    {cloud_info}  = {cloud_ip};{cloud_port};{cloud_path};{cloud_usr};{cloud_pwd}
                    {device_info} = {dev_type};{cuda_name};{verbose};{real_device_idx}
                    {data}        = {comm_round};{dev_model_name};{dev_model_filename};{local_epochs};{learning_rate};
                                    {train_type};{N_fcl};{coe};{kt};{batch_size};{num_rooms};{data_iid}
                    """
                    cloud_info = f"{cloud_ip};{cloud_port};{cloud_path};{cloud_usr};{cloud_pwd}"
                    device_info = f"{dev_type};{dev_cuda_name};{verbose};{i}"
                    data = (f"{comm_round};{dev_model_name};{dev_model_filename};{local_epochs};{learning_rate};"
                            f"{train_type};{N_fcl};{coe};{kt};{batch_size};{num_rooms};{data_iid}")

                    setup_message = f"{cloud_info},{device_info},{data}"
                    device_handler_list.append(
                        DeviceHandler(cloud_path=cloud_path, dev_idx=i, dev_host=dev_host, dev_port=dev_port,
                                      dev_usr=dev_usr, dev_pwd=dev_pwd, dev_path=dev_path,
                                      dev_model_filename=dev_model_filename, setup_message=setup_message,
                                      verbose=verbose)
                    )
                for idx, i in enumerate(avail_devs_idx_dict[mydict_key]):
                    device_handler_list[idx].start()
                if verbose:
                    print(f"[+] Wait until clients to finish their job")
                value = []
                for idx, i in enumerate(avail_devs_idx_dict[mydict_key]):
                    value.append(device_handler_list[idx].join())
            if verbose:
                print("\n[+] Joined all clients")


            local_weights = []
            local_distr = []
            for i in available_devices:
                dev_model_filename = f"dev_{i}.pth"
                local_weights.append(torch.load(path.join(cloud_path, dev_model_filename), map_location=torch.device(cloud_cuda_name)))
                local_distr.append(client_dist[i])

            if train_type == "mde_fedcl" or train_type == "mde_fedknow" or train_type == "mde_codeps":
                w_glob = aggregate_avg(local_weights=local_weights, local_distr=local_distr)
            elif train_type == "mde_loca":
                w_glob = net_glob.to(torch.device(cloud_cuda_name)).state_dict()
                w_glob = aggregate_cos(sigma=0.1, global_weights=w_glob, local_weights=local_weights, device=torch.device(cloud_cuda_name))

            torch.save(w_glob, path.join(cloud_path, f"global_weights.pth"))
            if comm_round >= 99:
                torch.save(w_glob, path.join(eval_path, f"global_weights.pth"))
            net_glob.load_state_dict(w_glob)

            scheduler.step()
            learning_rate = optimizer.param_groups[0]['lr']

            loss_test, metrics_test = global_test(model=net_glob, cuda_name=cloud_cuda_name, dataset_name=dataset_name,
                                                  seed=self.seed, num_rooms=num_rooms, nyuv2_test=False)
            with open(path.join(f"{base_dir}", log_file), 'a+') as logger:
                logger.write(f"{comm_round+1},{metrics_test[0].avg},{metrics_test[1].avg},{metrics_test[2].avg},{metrics_test[3].avg},{metrics_test[4].avg},{metrics_test[5].avg},{metrics_test[6].avg},{loss_test}\n")
            print(f"CommRound: {comm_round+1}; Deltas: d1 {metrics_test[4].avg:.4f} | d2 {metrics_test[5].avg:.4f} | d3 {metrics_test[6].avg:.4f}; Loss: {loss_test:.4f}")

            loss_test, metrics_test = global_test(model=net_glob, cuda_name=cloud_cuda_name, dataset_name=dataset_name,
                                                  seed=self.seed, num_rooms=num_rooms, nyuv2_test=True)
            with open(path.join(f"{base_dir}", nyuv2_log_file), 'a+') as logger:
                logger.write(f"{comm_round + 1},{metrics_test[0].avg},{metrics_test[1].avg},{metrics_test[2].avg},{metrics_test[3].avg},{metrics_test[4].avg},{metrics_test[5].avg},{metrics_test[6].avg},{loss_test}\n")
            print(f"NYUv2Test: {comm_round+1}; Deltas: d1 {metrics_test[4].avg:.4f} | d2 {metrics_test[5].avg:.4f} | d3 {metrics_test[6].avg:.4f}; Loss: {loss_test:.4f}\n")

            comm_round_time = time.time() - start_time
            comm_round_time_lst.append(comm_round_time)
            comm_round_time_avg = np.mean(comm_round_time_lst)

        print(f"Total time for experiment: {time.time() - total_time_start}")

        with open(path.join(f"{base_dir}", "time.csv"), 'a+') as logger:
            logger.write(f"{exp},{r},{time.time() - total_time_start},{comm_round_time_avg}\n")
        self.end_experiment(verbose=verbose)

    def end_experiment(self, verbose):
        if verbose:
            print("[+] Closing everything")
        device_handler_list = []
        with open(self.dev_cfg, 'r') as f:
            dt = json.load(f)
        for i in range(dt["num_devices"]):
            dev = dt[f"dev{i + 1}"]
            device_handler_list.append(
                DeviceHandler(dev_host=dev["hostname"], dev_port=dev["port"], setup_message="end", verbose=verbose)
            )

        if verbose:
            print("[+] Closing all clients...")

        for i in range(dt["num_devices"]):
            device_handler_list[i].start()

        if verbose:
            print("[+] Wait until clients close")

        for i in range(dt["num_devices"]):
            device_handler_list[i].join()

        if verbose:
            print("[+] Closed all clients")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cloud_cfg', type=str, default="configs/cloud_cfg_exp1.json",
                        help='Cloud configuration file name')
    parser.add_argument('--dev_cfg', type=str, default="configs/dev_cfg_exp1.json",
                        help='Device configuration file name')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    args = parser.parse_args()

    cloud = Cloud(cloud_cfg=args.cloud_cfg, dev_cfg=args.dev_cfg, seed=args.seed)
    cloud.federated_learning()
