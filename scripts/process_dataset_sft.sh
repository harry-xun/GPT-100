nohup python datasets/sft_process_dataset.py >> logs/process_dataset_sft_out.txt 2>&1 &
echo $! > logs/save_pid.txt