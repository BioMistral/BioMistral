import json
import torch
import argparse
from tqdm import tqdm
from pqdm.processes import pqdm
import torch.nn.functional as F
from datasets import load_dataset
from awq import AutoAWQForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM

# python PredictFewShot-Avg-Thread-AWQ.py --model_name="BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM"
# python PredictFewShot-Avg-Thread-AWQ.py --model_name="BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMV"

parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model_name", type=str, required=True, help="HuggingFace Hub / Local model name")
parser.add_argument("--use_exllama", action='store_true', help="Running with ExLLaMa v1")
parser.add_argument("--use_exllama_v2", action='store_true', help="Running with ExLLaMa v2")
args = parser.parse_args()
args = vars(args)

if args["use_exllama"] == True and args["use_exllama_v2"] == True:
    print("Cannot enable both ExLLaMa version at the same time!")
    exit(1)

if args["use_exllama"] == True:
    using_exllama = "exllama_v1"
elif args["use_exllama_v2"] == True:
    using_exllama = "exllama_v2"
else:
    using_exllama = ""

THREADS_NBR = 14

all_corpus = ["PubMedQA","MedQA","MedMCQA","MedQA-5_options"]
all_corpus = ["MMLU_" + subject for subject in ["clinical_knowledge","medical_genetics","anatomy","professional_medicine","college_biology","college_medicine"]] + all_corpus

def process(data):

    if "llama" in full_name.lower() or "medalpaca" in full_name.lower() or "doctor" in full_name.lower():
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

model = AutoAWQForCausalLM.from_quantized(full_name, fuse_layers=True, trust_remote_code=False, safetensors=True, use_exllama=args["use_exllama"], use_exllama_v2=args["use_exllama_v2"])
tokenizer = AutoTokenizer.from_pretrained(full_name, trust_remote_code=True)

for corpus in all_corpus:

    dataset = load_dataset("Project44/EnglishOnlyBioInstructQA", corpus)["test"]
    print(dataset)

    torch.set_default_device("cuda")

    for version in range(1,4):

        data_threads = []

        with torch.no_grad():
            
            for d in tqdm(dataset):

                inputs = tokenizer(d[f"prompt_no_answer_fewshot[{version}]"], return_tensors = "pt")

                input_ids = inputs["input_ids"].to("cuda")
                outputs = model.generate(input_ids, max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
                data_threads.append({"scores": outputs.scores[0].to("cpu"), "identifier": d["identifier"], "correct_letter": d[f"prompt_fewshot[{version}]"][-1], "classes": d["classes"]})

        data_batches = list(divide_chunks(data_threads, THREADS_NBR))

        all_thread_result = pqdm([{"data": db} for db in data_batches], process, n_jobs=THREADS_NBR, argument_type='kwargs')

        all_results = []
        for thread_result in all_thread_result:
            all_results.extend(thread_result)
        print("Total elements processed: ", len(all_results))

        f_out = open(f"./results_fewshot_avg_quantization/results_{short_model_name}+{using_exllama}_FewShot[{version}]_{corpus}_EN.json", 'w')
        json.dump(all_results, f_out)
        f_out.close()
