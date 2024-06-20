import os
import json
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from pqdm.processes import pqdm
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from sklearn.metrics import classification_report, accuracy_score

parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument("--base_model_name", type=str, required=True, help="HuggingFace Hub / Local model name")
parser.add_argument("--peft_name", type=str, required=True, help="HuggingFace Hub / Local model name")
parser.add_argument("--corpus", type=str, required=True, help="MMLU_clinical_knowledge / MMLU_medical_genetics, ...")
parser.add_argument("--few_shot_run", type=int, required=True, help="Few-shot run from 1 to 3 included")
args = parser.parse_args()
args = vars(args)

THREADS_NBR = 14

full_name = args["peft_name"]
short_model_name = full_name.split("/")[-1].replace("_","-")

base_model_name = args["base_model_name"]
short_base_model_name = base_model_name.split("/")[-1].replace("_","-")

if "llama" in full_name.lower() or "medalpaca" in full_name.lower() or "doctor" in full_name.lower():
    model = LlamaForCausalLM.from_pretrained(base_model_name, device_map="cuda", trust_remote_code=True)
    model.load_adapter(full_name)
    tokenizer = LlamaTokenizer.from_pretrained(full_name, trust_remote_code=True)
else:
    model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="cuda", trust_remote_code=True)
    model.load_adapter(full_name)
    tokenizer = AutoTokenizer.from_pretrained(full_name, trust_remote_code=True)

if os.path.isdir('./results_fewshot_avg_sft/') == False:
    os.makedirs('./results_fewshot_avg_sft/', exist_ok=True)

if args["corpus"] == "MMLU":
    in_corpus = ['MMLU_clinical_knowledge', 'MMLU_medical_genetics', 'MMLU_anatomy', 'MMLU_professional_medicine', 'MMLU_college_biology', 'MMLU_college_medicine']
elif args["corpus"] in ['PubMedQA', 'MedQA', 'MedMCQA', 'MedQA-5_options']:
    in_corpus = [args["corpus"]]
else:
    print("Error corpus name not found!")
    exit(1)

def process(data):

    results = []

    for current_data in tqdm(data):

        scores = F.softmax(current_data["scores"], dim=-1)
        max_len = scores.size(dim=1)
        top_k = torch.topk(scores, max_len)

        probs = top_k.values[0].cpu()
        token_str = [tokenizer.decode(top_k.indices[0].cpu()[i]) for i in range(max_len)]
        
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

for corpus_name in in_corpus:

    dataset = load_dataset(f"Project44/EnglishOnlyBioInstructQA", corpus_name)
    print(dataset)

    dataset = dataset[f"test"]

    torch.set_default_device("cuda")

    version = args["few_shot_run"]

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

    acc = accuracy_score(
        [r["correct_letter"] for r in all_results],
        [r["best_prediction"] for r in all_results]
    )
    print(acc)

    with open(f"./results_fewshot_avg_sft/results_{short_model_name}_FewShot[{version}]_{corpus_name}_english.json", 'w') as f:
        json.dump(all_results, f, indent=4)
