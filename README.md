# (CS322 Final Project) Repo for LLM-based APR approach with Python pretraining

This README will detail which commands to run to reproduce results. Note that most of these commands will be nohup Python commands. The most recently executed PID will be saved to `logs/save_pid.txt`, and that process can be killed by executing the script below.
```
bash scripts/kill_process.sh
```

# Initialize
Install required packages and create required directories. 
```
pip install torch==2.8.0
pip install -r requirements.txt
mkdir logs
mkdir generations
```

# Datasets

## Pretraining - CodeParrot

Run the nohup script below to stream the [CodeParrot](https://huggingface.co/datasets/transformersbook/codeparrot) dataset, filter based on Python packages, split into train/test splits and push to `Harryxun/GPT-100-dataset` HuggingFace dataset repository. 

```
bash scripts/process_dataset.sh
```
Log output directory: `./logs/process_dataset_out.txt`.

## SFT - FixEval

Clone the [FixEval](https://github.com/mahimanzum/FixEval) repository into the GPT-100 repository. 
```
git clone https://github.com/mahimanzum/FixEval.git
```

Run the script below to filter the FixEval Python examples based on the bug type (default: `Runtime Errors`), split into train/test based on the given FixEval test split, and push to `Harryxun/GPT-100-sft-dataset` HuggingFace dataset repository. 

```
bash scripts/process_datset_sft.sh
```
Log output directory: `./logs/process_dataset_sft_out.txt`.

## SFT - Python-Bugs

Run the script below to load/preprocess the [Python-Bugs](https://huggingface.co/datasets/Muennighoff/python-bugs) dataset and push to `Harryxun/GPT-100-sft-dataset-small` HuggingFace dataset repository. 

```
bash scripts/process_datset_sft2.sh
```
Log output directory: `./logs/process_dataset_sft2_out.txt`.

# Pretraining

Run the script below to execute pretraining and push to `Harryxun/llama-pretrained` model repository. (Estimated Time: 24+ hours. 55+ GB of VRAM required.) 

```
bash scripts/pretrain.sh
```
Log output directory: `./logs/pretrain_out.txt`.

Model checkpoint directory: `./llama-pretrained`.

# SFT (LoRA)

## FixEval

Run the script below to execute SFT for FixEval and push to `Harryxun/llama-sft` model repository. (Estimated Time: 20 hours. 55+ GB of VRAM required.) 

```
bash scripts/sft.sh
```
Log output directory: `./logs/sft.txt`.

Model checkpoint directory: `./llama-sft`.

## Python-Bugs

Run the script below to execute SFT for FixEval and push to `Harryxun/llama-sft2` model repository. (Estimated Time: 20 hours. 55+ GB of VRAM required.) 

```
bash scripts/sft2.sh
```
Log output directory: `./logs/sft2.txt`.

Model checkpoint directory: `./llama-sft2`.

# Evaluation

Run the Python file below to run evaluation on a selected model checkpoint/path (pretraining or sft). Manual edit of the `prompt` variable is required.
```
python sft_eval.py
```

## Evaluation (FixEval Test Suite)

Run the Python file below to use the SFT model to generate outputs and parse them according to the FixEval testing suite requirements. 

```
python generate_outputs.py
python generate_process.py
```

Run the script below to run the testing suite on the generated model outputs. 

```
bash scripts/eval.sh
```
Log output directory: `./logs/eval_out.txt`.

## Evaluation (Python-Bugs)

Run the Python file below to use the SFT model to generate outputs from the Python-Bugs test split.

```
python generate_outputs2.py
```

Run the Python file below to calculate the exact match rate of both datasets and plot the Levenshtein distance distribution. (Make sure to run both `generate_outputs.py` and `generate_outputs2.py` beforehand.) Change `CUTLINE` variable to alter Levenshtein distance cutline to be considered correct. 

```
python evaluate.py
```

# Acknowledgement

This repo benefits from PyTorch, Transformers, TRL, PEFT, and FixEval. 