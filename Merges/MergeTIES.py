import os

# python MergeTIES.py

MODEL_NAME = "BioMistral-7B-ties"
yaml_config = """
models:
  - model: mistralai/Mistral-7B-Instruct-v0.1
  - model: Project44/BioMistral-7B-0.1-PubMed-V2
    parameters:
      density: 0.5
      weight: 0.5
merge_method: ties
base_model: mistralai/Mistral-7B-Instruct-v0.1
parameters:
  normalize: true
dtype: bfloat16
"""

with open('config_ties.yaml', 'w', encoding="utf-8") as f:
    f.write(yaml_config)

os.system("mergekit-yaml config_ties.yaml BioMistral-7B-mistral7instruct-ties --copy-tokenizer --cuda --low-cpu-memory --trust-remote-code")
