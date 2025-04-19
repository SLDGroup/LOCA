from os import path
import os
import argparse
from utils.communication import receive_msg, send_msg, close_connection
from utils.train_test import local_training
from utils.general_utils import get_hw_info,seed_everything
import socket
import torch

class Device:
    def __init__(self, connection, exp, r, dev_idx, seed, verbose=False):
        seed_everything(seed=seed)
        self.seed = seed
        self.connection = connection
        self.exp = exp
        self.r = r
        self.dev_idx = dev_idx
        self.verbose = verbose

    def run(self):
        """
        Main method for the Device. The Cloud sends a message to the Device in the following format:

        setup_message = {cloud_info},{device_info},{data}

        where

        {cloud_info}  = {target_ip};{target_port};{target_path};{target_usr};{target_pwd}
        {device_info} = {hw_type};{cuda_name};{verbose}
        {data}        = {comm_round};{model_name};{filename};{local_epochs};{learning_rate};{train_type};{save_opt};{labeled}
        """
        # Step 1. Receive setup message from Cloud.
        setup_message = receive_msg(self.connection, verbose=self.verbose)

        # Exit if there is no message.
        if setup_message is None:
            print(f"[!] No message received from the Cloud on device {self.dev_idx}.")
            return False

        if setup_message == "end":
            close_connection(connection=self.connection, verbose=self.verbose)
            return True

        cloud_info, device_info, data = setup_message.split(',')

        hw_type, cuda_name, self.verbose, real_device_idx = device_info.split(';')
        real_device_idx = int(real_device_idx)
        dev_pwd, dev_usr, dev_path, eval_dev_path = get_hw_info(hw_type)
        os.makedirs(dev_path, exist_ok=True)
        self.verbose = bool(int(self.verbose == 'True'))

        (comm_round, model_name, model_filename, local_epochs, learning_rate, train_type, N_fcl, coe, kt,
         batch_size, num_rooms, data_iid) = data.split(';')
        comm_round = int(comm_round)
        N_fcl = int(N_fcl)
        coe = float(coe)
        kt = int(kt)
        local_epochs = int(local_epochs)
        learning_rate = float(learning_rate)
        batch_size = int(batch_size)

        data_iid = bool(int(data_iid == 'True'))

        model_path = path.join(dev_path, model_filename)
        buf_path = path.join(dev_path, f"buffer_{real_device_idx}.pkl")

        # Step 2. Synch with Cloud with msg "done_setup"
        send_msg(connection=self.connection, msg="done_setup", verbose=self.verbose)


        # Step 3. Train the model
        train_loss, train_metrics, train_time = \
            local_training(model_name=model_name, train_type=train_type, model_path=model_path,
                           cuda_name=cuda_name, learning_rate=learning_rate, local_epochs=local_epochs,
                           dev_idx=real_device_idx, verbose=self.verbose, seed=self.seed, comm_round=comm_round,
                           N_fcl=N_fcl, coe=coe, kt=kt, batch_size=batch_size, buf_path=buf_path,
                           num_rooms=num_rooms, data_iid=data_iid)
            
        # Step 4. Sync with Cloud with "done_training" message
        send_msg(connection=self.connection, msg=f"done_training;{train_time}", verbose=self.verbose)
        
        close_connection(connection=self.connection, verbose=self.verbose)
        return False


def start_device(host, port, dev_idx, exp, r, seed, verbose=False):
    """
    Starts the device, handles incoming connections, and records power and temperature (optional).

    Parameters:
        host (str): Hostname or IP to bind the device to.
        port (int): Port to bind the device to.
        dev_idx (int): Index of the device.
        exp (int): Experiment number.
        r (int): Run number.
        verbose (bool, optional): If True, displays status messages during execution. Defaults to False.

    Returns:
        None
    """

    os.makedirs("logs", exist_ok=True)

    # Create a socket object
    soc = socket.socket()
    # Set the socket to reuse the address
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Bind the socket to the host and port
    soc.bind((host, port))

    if verbose:
        print(f"[+] Socket created on device {dev_idx}")

    try:
        while True:
            # Listen for incoming connections
            soc.listen(5)
            if verbose:
                print(f"[+] Device {dev_idx} is listening on host:port {host}:{port}")
            # Accept a connection
            connection, _ = soc.accept()
            # Create a Device instance with the accepted connection
            device = Device(connection=connection, exp=exp, r=r, dev_idx=dev_idx, seed=seed, verbose=verbose)
            # Run the device
            if device.run():
                break
    except BaseException as e:
        print(f'[!] Device {dev_idx} ERROR: {e} Socket closed due to no incoming connections or error.')
    finally:
        # Close the socket
        soc.close()


parser = argparse.ArgumentParser(description="Device arguments")
parser.add_argument('--host', default="127.0.0.1", help='IP address of the device')
parser.add_argument('--port', type=int, default=10000, help='Port for the device to listen on')
parser.add_argument('--dev_idx', type=int, default=0, help='Index of the device')
parser.add_argument('--exp', type=int, default=0,  help='Experiment number')
parser.add_argument('--r', type=int, default=0, help='Run number')
parser.add_argument('--seed', type=int, default=42, help='Seed number')
parser.add_argument('--verbose', action='store_true', help='If verbose or not')
args = parser.parse_args()

start_device(host=args.host, port=args.port, dev_idx=args.dev_idx, exp=args.exp, r=args.r, seed=args.seed,
             verbose=args.verbose)
