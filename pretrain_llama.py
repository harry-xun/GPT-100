from datasets import load_dataset

ds = load_dataset("Harryxun/GPT-100-pretrain", split="chunk_1")
print(ds)
print(ds[0]) 
print(ds[0].keys()) 
print(ds[0]["content"][:500]) 
