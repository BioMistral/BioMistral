import json
import torch
import argparse
from tqdm import tqdm
from pqdm.processes import pqdm
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM

parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model_name", type=str, required=True, help="HuggingFace Hub / Local model name")
args = parser.parse_args()
args = vars(args)

THREADS_NBR = 14

def process(data):

    if "llama" in full_name.lower() :
        tokenizer_thread = LlamaTokenizer.from_pretrained(full_name, trust_remote_code=True)
    else:
        tokenizer_thread = AutoTokenizer.from_pretrained(full_name, trust_remote_code=True)
    
    results = []

    for current_data in tqdm(data):

        scores = F.softmax(current_data["scores"], dim=-1)
        max_len = scores.size(dim=1)
        top_k = torch.topk(scores, max_len)

        probs = top_k.values[0].cpu()
        token_str = [tokenizer_thread.decode(top_k.indices[0].cpu()[i]) for i in range(max_len)]
        
        kv = [{"token_str": ts, "probs": tp.item()} for tp, ts in zip(probs, token_str) if ts in current_data["classes"]]

        predictions = {}
        for pair in kv:            
            if pair["token_str"] not in predictions:
                predictions[pair["token_str"]] = pair["probs"]

        results.append({
            "identifier": current_data["identifier"],
            "correct_letter": current_data["correct_letter"],
            "predictions": predictions,
            "best_prediction": token_str[0],
        })
    
    return results

def divide_chunks(l, n):
    output_chunks = []
    for i in range(0, len(l), n):  
        output_chunks.append(l[i:i + n])
    return output_chunks

full_name = args["model_name"].rstrip("/")
short_model_name = full_name.split("/")[-1].replace("_","-")
print(short_model_name)

if "llama" in full_name.lower():
    model = LlamaForCausalLM.from_pretrained(full_name, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    tokenizer = LlamaTokenizer.from_pretrained(full_name, trust_remote_code=True)
else:
    model = AutoModelForCausalLM.from_pretrained(full_name, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(full_name, trust_remote_code=True)

dataset = load_dataset("Project44/TruthfulQA")["test"]
print(dataset)

torch.set_default_device("cuda")

data_threads = []

with torch.no_grad():
    
    for d in tqdm(dataset):

        inputs = tokenizer(d[f"prompt_no_answer_bis"], return_tensors = "pt")

        input_ids = inputs["input_ids"].to("cuda")
        outputs = model.generate(input_ids, max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
        data_threads.append({"scores": outputs.scores[0].to("cpu"), "identifier": d["identifier"], "correct_letter": d[f"prompt_bis"][-1], "classes": d["classes"]})

data_batches = list(divide_chunks(data_threads, THREADS_NBR))

all_thread_result = pqdm([{"data": db} for db in data_batches], process, n_jobs=THREADS_NBR, argument_type='kwargs')

all_results = []
for thread_result in all_thread_result:
    all_results.extend(thread_result)
print("Total elements processed: ", len(all_results))

f_out = open(f"./results_TruthfulQA_PromptMeditron/results_{short_model_name}_ZeroShot_TruthfulQA_EN.json", 'w')
json.dump(all_results, f_out)
f_out.close()
