from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


DATASET_PATH = "Harryxun/stf_run"
SEED = 42

# MODEL_PATH = "./llama-pretrained-1/checkpoint-83824"
# MODEL_PATH = "./llama-pretrained/checkpoint-10000"
MODEL_PATH = "./llama-sft2/checkpoint-1500"
device = "cuda" 


# load pretrained/sft model/tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
model.eval()


# example prompt
prompt = """\
def __init__(self, owner=None):
    owner.owner = owner
    self.editing = False

    QTreeWidget.__init__(self, owner)
    self.setColumnCount(3)
    self.setHeaderLabels([_("Address"), _("Label"), _("Used")])
    self.setIndentation(0)

    self.hide_used = True
    self.setColumnHidden(2, True)\n\n### FIXED CODE ###\n\n
"""


inputs = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
        repetition_penalty=1.1
    )

print("Model:\n", tokenizer.decode(out[0], skip_special_tokens=True))
