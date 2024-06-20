import os

yaml_config = """
slices:
  - sources:
      - model: berkeley-nest/Starling-LM-7B-alpha
        layer_range: [0, 32]
      - model: Project44/BioMistral-7B-0.1-PubMed-V2
        layer_range: [0, 32]
merge_method: slerp
base_model: berkeley-nest/Starling-LM-7B-alpha
parameters:
  t:
    - filter: self_attn
      value: [0, 0.5, 0.3, 0.7, 1]
    - filter: mlp
      value: [1, 0.5, 0.7, 0.3, 0]
    - value: 0.5
dtype: bfloat16
"""

with open('config_starling.yaml', 'w', encoding="utf-8") as f:
    f.write(yaml_config)

###########################################

yaml_config = """
slices:
  - sources:
      - model: teknium/OpenHermes-2-Mistral-7B
        layer_range: [0, 32]
      - model: Project44/BioMistral-7B-0.1-PubMed-V2
        layer_range: [0, 32]
merge_method: slerp
base_model: teknium/OpenHermes-2-Mistral-7B
parameters:
  t:
    - filter: self_attn
      value: [0, 0.5, 0.3, 0.7, 1]
    - filter: mlp
      value: [1, 0.5, 0.7, 0.3, 0]
    - value: 0.5
dtype: bfloat16
"""

with open('config_OpenHermes.yaml', 'w', encoding="utf-8") as f:
    f.write(yaml_config)

###########################################

yaml_config = """
slices:
  - sources:
      - model: HuggingFaceH4/zephyr-7b-beta
        layer_range: [0, 32]
      - model: Project44/BioMistral-7B-0.1-PubMed-V2
        layer_range: [0, 32]
merge_method: slerp
base_model: HuggingFaceH4/zephyr-7b-beta
parameters:
  t:
    - filter: self_attn
      value: [0, 0.5, 0.3, 0.7, 1]
    - filter: mlp
      value: [1, 0.5, 0.7, 0.3, 0]
    - value: 0.5
dtype: bfloat16
"""

with open('config_zephyr.yaml', 'w', encoding="utf-8") as f:
    f.write(yaml_config)

###########################################

yaml_config = """
slices:
  - sources:
      - model: mistralai/Mistral-7B-Instruct-v0.1
        layer_range: [0, 32]
      - model: Project44/BioMistral-7B-0.1-PubMed-V2
        layer_range: [0, 32]
merge_method: slerp
base_model: mistralai/Mistral-7B-Instruct-v0.1
parameters:
  t:
    - filter: self_attn
      value: [0, 0.5, 0.3, 0.7, 1]
    - filter: mlp
      value: [1, 0.5, 0.7, 0.3, 0]
    - value: 0.5
dtype: bfloat16
"""

with open('config.yaml', 'w', encoding="utf-8") as f:
    f.write(yaml_config)

os.system("mergekit-yaml config.yaml BioMistral-7B-slerp --copy-tokenizer --cuda --low-cpu-memory --trust-remote-code")
os.system("mergekit-yaml config_starling.yaml BioMistral-7B-Starling-slerp --copy-tokenizer --cuda --low-cpu-memory --trust-remote-code")
os.system("mergekit-yaml config_OpenHermes.yaml BioMistral-7B-OpenHermes-slerp --copy-tokenizer --cuda --low-cpu-memory --trust-remote-code")
os.system("mergekit-yaml config_zephyr.yaml BioMistral-7B-zephyr-beta-slerp --copy-tokenizer --cuda --low-cpu-memory --trust-remote-code")
