# git clone https://huggingface.co/mistralai/Mistral-7B-v0.1

import os
from transformers import AutoTokenizer, AutoModelForCausalLM

path = "./models/mistralai_Mistral-7B-Instruct-v0.1/"

os.makedirs(path, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer.save_pretrained(path)

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model.save_pretrained(path)
