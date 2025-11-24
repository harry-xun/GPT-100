nohup python sft.py >> logs/sft_out.txt 2>&1 &
echo $! > logs/save_pid.txt