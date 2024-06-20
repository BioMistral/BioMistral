from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'Project44/BioMistral-7B-0.1-PubMed-V2'

q_group_size = 128
w_bit = 4
version = "GEMV"
# version = "GEMM"

quant_path = f"BioMistral-7B-Instruct-AWQ-QGS{q_group_size}-W{w_bit}-{version}"
quant_config = {"zero_point": True, "q_group_size": q_group_size, "w_bit": w_bit, "version": version}

# Load model
# NOTE: pass safetensors=True to load safetensors
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
