import os

# python MergeDARE.py

MODEL_NAME = "BioMistral-7B-dare"
yaml_config = """
models:
  - model: mistralai/Mistral-7B-Instruct-v0.1
    # No parameters necessary for base model
  - model: Project44/BioMistral-7B-0.1-PubMed-V2
    parameters:
      density: 0.5
      weight: 0.5
merge_method: dare_ties
base_model: mistralai/Mistral-7B-Instruct-v0.1
parameters:
  int8_mask: true
dtype: bfloat16
"""

# Save config as yaml file
with open('config_dare.yaml', 'w', encoding="utf-8") as f:
    f.write(yaml_config)

os.system("mergekit-yaml config_dare.yaml BioMistral-7B-mistral7instruct-dare --copy-tokenizer --cuda --low-cpu-memory --trust-remote-code")
