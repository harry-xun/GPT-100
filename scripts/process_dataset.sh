nohup python datasets/pretrain_process_dataset.py >> logs/process_dataset_out.txt 2>&1 &
echo $! > logs/save_pid.txt