nohup python pretrain.py >> logs/pretrain_out.txt 2>&1 &
echo $! > logs/save_pid.txt