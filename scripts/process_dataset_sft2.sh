nohup python datasets/sft_process_dataset2.py >> logs/process_dataset_sft2_out.txt 2>&1 &
echo $! > logs/save_pid.txt