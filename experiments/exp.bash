#!/bin/bash

trap "kill 0" EXIT

OLDIFS=$IFS

cuda_limit_per_gpu=3
num_devices=10 # number of simulated devices at the same time

declare -a experiment_configs=(
#########################################################################################################################
# experiment 0| runs 1| device_cuda_name 2| dev_local_epochs (k1) 3| kt 4| train_type 5| bs 6 | N_fcl 7 |
# num_rooms 8| cuda_order 9| data_iid 10|

  "1000 3 cuda:0 1 1 mde_fedcl   16 1  3   0123 false"
  "1001 3 cuda:0 1 1 mde_fedcl   16 1  4   0123 false"
  "1002 3 cuda:0 1 1 mde_fedcl   16 1  345 0123 false"

  "1100 3 cuda:0 1 1 mde_fedcl   16 10 3   0123 false"
  "1101 3 cuda:0 1 1 mde_fedcl   16 10 4   0123 false"
  "1102 3 cuda:0 1 1 mde_fedcl   16 10 345 0123 false"

  "1200 3 cuda:0 1 1 mde_fedknow 16 1  3   0123 false"
  "1201 3 cuda:0 1 1 mde_fedknow 16 1  4   0123 false"
  "1202 3 cuda:0 1 1 mde_fedknow 16 1  345 0123 false"

  "1300 3 cuda:0 1 1 mde_codeps  16 1  3   0123 false"
  "1301 3 cuda:0 1 1 mde_codeps  16 1  4   0123 false"
  "1302 3 cuda:0 1 1 mde_codeps  16 1  345 0123 false"

  "1400 3 cuda:0 1 1 mde_loca    16 1  3   0123 false"
  "1401 3 cuda:0 1 1 mde_loca    16 1  4   0123 false"
  "1402 3 cuda:0 1 1 mde_loca    16 1  345 0123 false"

# experiment 0| runs 1| device_cuda_name 2| dev_local_epochs (k1) 3| kt 4| train_type 5| bs 6 | N_fcl 7 |
# num_rooms 8| cuda_order 9| data_iid 10|
)

#### Configs ####
PORT=10000
PORT_INCR=50
seeds=(42 182 342)
comm_rounds=100
verbose='false'
learning_rate=0.001
model_name="guidedepth-s"
num_users=50 # total number of devices (device indexes)
dataset_name="nyuv2"
dev_fraction=0.2

#### FedCL ####
coe=0.5


for experiment_config in "${experiment_configs[@]}"
do
  read -ra exp_config <<< "$experiment_config"
  experiment="${exp_config[0]}"
  runs="${exp_config[1]}"
  device_cuda_name="${exp_config[2]}"
  dev_local_epochs="${exp_config[3]}"
  kt="${exp_config[4]}"
  train_type="${exp_config[5]}"
  batch_size="${exp_config[6]}"
  N_fcl="${exp_config[7]}"
  num_rooms="${exp_config[8]}"
  cuda_order="${exp_config[9]}"
  data_iid="${exp_config[10]}"

  read -ra cuda_devices <<< "$(echo $cuda_order | sed 's/\(.\)/\1 /g')"
  for (( run=1; run<=runs; run++ ))
  do
    total_cuda=1
    files_number="files_${experiment}_${run}"
    seed=${seeds[$((run-1))]}
    for i in $(seq 0 $((num_devices-1))) # number of simulated devices at the same time
    do
      if [ $total_cuda -gt $((cuda_limit_per_gpu*3)) ]
      then
          cuda_names[$i]="cuda:${cuda_devices[3]}"
      elif [ $total_cuda -gt $((cuda_limit_per_gpu*2)) ]
      then
          cuda_names[$i]="cuda:${cuda_devices[2]}"
      elif [ $total_cuda -gt $((cuda_limit_per_gpu)) ]
      then
          cuda_names[$i]="cuda:${cuda_devices[1]}"
      else
          cuda_names[$i]="cuda:${cuda_devices[0]}"
      fi
      ports[$i]=$((PORT+i))
      total_cuda=$((total_cuda+1))
    done

    cloud_config_filename="cloud_cfg_exp${experiment}_run${run}.json"
    dev_config_filename="dev_cfg_exp${experiment}_run${run}.json"
    exec_name="exec_exp${experiment}_run${run}.bash"

    echo "${experiment}", "${run}"
    python generate_configs.py \
    --cloud_config_filename "${cloud_config_filename}" \
    --dev_config_filename "${dev_config_filename}" \
    --cloud_cuda_name "${device_cuda_name}" \
    --exec_name "${exec_name}" \
    --experiment "${experiment}" \
    --run "${run}" \
    --seed "${seed}" \
    --model_name "${model_name}" \
    --train_type "${train_type}" \
    --comm_rounds "${comm_rounds}" \
    --verbose "${verbose}" \
    --num_devices "${num_devices}" \
    --dataset_name "${dataset_name}" \
    --num_users "${num_users}" \
    --dev_fraction "${dev_fraction}" \
    --learning_rate "${learning_rate}" \
    --batch_size "${batch_size}" \
    --files_number "${files_number}" \
    --dev_hw_type "${files_number}" \
    --ports "${ports[*]}" \
    --cuda_names "${cuda_names[*]}" \
    --dev_local_epochs "${dev_local_epochs}" \
    --N_fcl "${N_fcl}" \
    --coe "${coe}" \
    --kt "${kt}" \
    --num_rooms "${num_rooms}"\
    --data_iid "${data_iid}"

    PORT=$((PORT+PORT_INCR))
  done
done
IFS=$OLDIFS