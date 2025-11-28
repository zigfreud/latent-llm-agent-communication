# Teste Rápido do Tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")

texto = "def soma(a,b):"

# As is before:
tokens_antigo = tokenizer(texto, return_tensors="pt").input_ids[0]
print(f"Antigo (token 0): {tokens_antigo[0]} -> {tokenizer.decode(tokens_antigo[0])}")

# As it should be:
tokens_novo = tokenizer(texto, return_tensors="pt", add_special_tokens=False).input_ids[0]
print(f"Novo (token 0):   {tokens_novo[0]} -> {tokenizer.decode(tokens_novo[0])}")