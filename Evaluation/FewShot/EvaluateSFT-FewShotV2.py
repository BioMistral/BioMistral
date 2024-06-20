import os
import json
import statistics

from sklearn.metrics import classification_report, accuracy_score

models_results = {}

bad_corpus_names = ["MMLU_" + m for m in ["clinical_knowledge","medical_genetics","anatomy","professional_medicine","college_biology","college_medicine"]] + ["MedQA","MedQA-5_options","PubMedQA","MedMCQA"]
good_corpus_name = [gcn.replace("_","-") for gcn in bad_corpus_names]

for file_name in os.listdir("./results_fewshot_avg_sft/"):

    if "_FewShot" not in file_name:
        continue
    
    new_file_name = file_name
    for mmlu_c in bad_corpus_names:
        new_file_name = new_file_name.replace(mmlu_c, mmlu_c.replace("_","-"))
    splitted = new_file_name.replace(".json","").split("_")
    model_name = splitted[1]
    shot_mode, id_shot = splitted[2].split("[")
    id_shot = id_shot.replace("]","")
    corpus = splitted[3]
    lang = splitted[4]

    f = open(f"./results_fewshot_avg_sft/{file_name}")
    data = json.load(f)
    f.close()

    refs = []
    preds = []

    for d in data:
        refs.append(d["correct_letter"])
        preds.append(d["best_prediction"])

    acc = accuracy_score(refs, preds)
    print(acc)

    if model_name not in models_results:
        models_results[model_name] = {}

    if corpus not in models_results[model_name]:
        models_results[model_name][corpus] = []

    models_results[model_name][corpus].append(acc)

with open("./merged_results_FewShot-SFT-V2_Avg-3-Runs.json", 'w') as f:
    json.dump(models_results, f, indent=4)

averaged_results = {}
std_deviation_results = {}

for model_name in models_results:

    if model_name not in averaged_results:
        averaged_results[model_name] = {}
        std_deviation_results[model_name] = {}

    for corpus_n in models_results[model_name]:

        if len(models_results[model_name][corpus_n]) < 3:
            print("Warning not enough runs for: ", model_name, " - ", corpus_n, " : ", 3-len(models_results[model_name][corpus_n]))

        avg_score = sum(models_results[model_name][corpus_n]) / len(models_results[model_name][corpus_n])
        averaged_results[model_name][corpus_n] = avg_score
        std_deviation_results[model_name][corpus_n] = statistics.pstdev([v*100 for v in models_results[model_name][corpus_n]])

with open("./averaged_results_FewShot-SFT-V2_Avg-3-Runs.json", 'w') as f:
    json.dump(averaged_results, f, indent=4)

with open("./std_deviation_results_FewShot-SFT-V2_Avg-3-Runs.json", 'w') as f:
    json.dump(std_deviation_results, f, indent=4)

line = " & MMLU & " + " & ".join([m_name.replace("-"," ") for m_name in good_corpus_name]) + " \\\\"
print(line)

for model_name in sorted(list(models_results.keys())):

    values_out = [(corpus_name, averaged_results[model_name][corpus_name]*100, std_deviation_results[model_name][corpus_name]) if corpus_name in averaged_results[model_name] else (corpus_name, 0, 0) for corpus_name in good_corpus_name]
    values_out_mmlu = [v for c, v, std_v in values_out if "MMLU" in c]
    avg_mmlu = sum(values_out_mmlu) / len(values_out_mmlu)

    values_std_v_mmlu = [std_v for c, v, std_v in values_out if "MMLU" in c]
    avg_mmlu_std_v = sum(values_std_v_mmlu) / len(values_std_v_mmlu)

    formatted_values_out = ["{:.1f}".format(v) + " \\scalebox{1.0}{\\tiny {" + "±{:.1f}".format(std_v) + "}}"  for c, v, std_v in values_out]

    line = f"{model_name} & " + "{:.1f}".format(avg_mmlu) + " \\scalebox{1.0}{\\tiny {" + "±{:.1f}".format(avg_mmlu_std_v) + "}}" + " & " + " & ".join(formatted_values_out) + " \\\\"
    print(line)
