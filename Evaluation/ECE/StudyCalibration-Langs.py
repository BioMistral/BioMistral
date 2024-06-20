import os
import json

import numpy as np

def expected_calibration_error(samples, true_labels, M=5):
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(samples, axis=1)
    predicted_label = np.argmax(samples, axis=1)

    accuracies = predicted_label==true_labels

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece.item()

bad_corpus_names = ["MMLU_" + m for m in ["clinical_knowledge","medical_genetics","anatomy","professional_medicine","college_biology","college_medicine"]] + ["MedQA","MedQA-5_options","PubMedQA","MedMCQA"]
good_corpus_name = [gcn.replace("_","-") for gcn in bad_corpus_names]

DIR = "./results/"

allowed_model = ["BioMistral-7B-0.1-PubMed-V2"]

scores = {}

for file_name in os.listdir(DIR):

    file_path = f"{DIR}/{file_name}"

    new_file_name = file_name
    for mmlu_c in bad_corpus_names:
        new_file_name = new_file_name.replace(mmlu_c, mmlu_c.replace("_","-"))

    splitted = new_file_name.replace(".json","").split("_")
    model_name = splitted[1]
    shot_mode, id_shot = splitted[2].split("[")
    id_shot = id_shot.replace("]","")
    corpus = splitted[3]
    lang = splitted[4]

    if model_name not in allowed_model:
        continue

    f_in = open(file_path,"r")
    data = json.load(f_in)
    f_in.close()

    samples = []
    true_labels = []

    letters = sorted(list(data[0]["predictions"].keys()))

    for d in data:
        samples.append([d["predictions"][l] for l in letters])
        true_labels.append(letters.index(d["correct_letter"]))

    if lang not in scores:
        scores[lang] = {}
    
    scores[lang][corpus] = expected_calibration_error(
        np.asarray(samples),
        np.asarray(true_labels)
    )*100

f_out = open(f"./scores-langs-BioMistral7BInstruct.json", 'w')
json.dump(scores, f_out, indent=4)
f_out.close()
