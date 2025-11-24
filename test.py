from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Harryxun/llama-pretrained")
s = "def foo(x):\n    return x + 1"
ids = tok(s)["input_ids"]
print(tok.decode(ids))