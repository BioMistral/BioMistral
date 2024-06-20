import os
import json
import statistics

from sklearn.metrics import accuracy_score

def EnsembleModels(MODEL_1, MODEL_2):

    models_results = {}

    bad_corpus_names = ["MedQA-5_options","MedQA","PubMedQA","MedMCQA"] + ["MMLU_" + m for m in ["clinical_knowledge","medical_genetics","anatomy","professional_medicine","college_biology","college_medicine"]]

    all_langs = []

    emsemble_models = [
        MODEL_1,
        MODEL_2
    ]

    ALL_LETTERS = ["A","B","C","D","E"]

    scores = {m_name: {index: {} for index in range(1,4)} for m_name in emsemble_models}
    references = {m_name: {index: {} for index in range(1,4)} for m_name in emsemble_models}

    all_corpus = []

    for current_shot in range(1,4):

        for file_name in os.listdir("./results_fewshot_avg_translated/"):

            if "_FewShot" not in file_name or "english" not in file_name or f"[{current_shot}]" not in file_name:
                continue


            new_file_name = file_name
            for mmlu_c in bad_corpus_names:
                new_file_name = new_file_name.replace(mmlu_c, mmlu_c.replace("_","-"))
            splitted = new_file_name.replace(".json","").split("_")
            model_name = splitted[1]

            if model_name not in list(scores.keys()):
                continue
            # print(file_name)

            shot_mode, id_shot = splitted[2].split("[")
            id_shot = id_shot.replace("]","")
            corpus = splitted[3]
            lang = splitted[4]

            f = open(f"./results_fewshot_avg_translated/{file_name}","r")
            data = json.load(f)
            f.close()

            # refs = []
            # preds = []

            for d in data:
                
                vec = []
                
                for letter in ALL_LETTERS:
                    if letter in list(d["predictions"].keys()):
                        vec.append(d["predictions"][letter])
                
                if corpus not in scores[model_name][current_shot]:
                    scores[model_name][current_shot][corpus] = []
                    references[model_name][current_shot][corpus] = []
                    all_corpus.append(corpus)

                scores[model_name][current_shot][corpus].append(vec)
                references[model_name][current_shot][corpus].append(d["correct_letter"])

    all_corpus = ["MedQA","MedQA-5_options","PubMedQA","MedMCQA"]
    all_corpus = ["MMLU_" + subject for subject in ["clinical_knowledge","medical_genetics","anatomy","professional_medicine","college_biology","college_medicine"]] + all_corpus

    average_results = []

    for corpus_name in all_corpus:

        corpus_name = corpus_name.replace("_","-")

        merged_scores = {index: [] for index in range(1,4)}
        predictions = {index: [] for index in range(1,4)}

        all_accuracies = []

        for index_shot in range(1,4):

            if references[MODEL_1][index_shot][corpus_name] != references[MODEL_2][index_shot][corpus_name]:
                print("Error: references not aligned!")
                exit(1)
            
            for m1_scores, m2_scores in zip(scores[MODEL_1][index_shot][corpus_name], scores[MODEL_2][index_shot][corpus_name]):

                new_vec = [v1 + v2 for v1, v2 in zip(m1_scores, m2_scores)]

                merged_scores[index_shot].append(new_vec)

                max_val = max(new_vec)
                index_val = new_vec.index(max_val)
                max_val_letter = ALL_LETTERS[index_val]
                predictions[index_shot].append(max_val_letter)
            
            shot_acc = accuracy_score(references[MODEL_1][index_shot][corpus_name], predictions[index_shot])*100
            all_accuracies.append(shot_acc)

        avg_acc = sum(all_accuracies) / len(all_accuracies)
        avg_acc = "%.2f" % avg_acc

        shots_std = statistics.pstdev([v for v in all_accuracies])
        shots_std = "%.1f" % shots_std

        average_results.append(avg_acc + " \\scalebox{1.0}{\\tiny {±" + shots_std + "}}")
        
    row = " & ".join(all_corpus)
    print(row)
    row = " & ".join(average_results)
    print(row)

print()
print("Mistral-7B-Instruct-v0.1", " --- ", "BioMistral-7B-0.1-PubMed-V2")
EnsembleModels("Mistral-7B-Instruct-v0.1", "BioMistral-7B-0.1-PubMed-V2")
print()

print("Mistral-7B-v0.1", " --- ", "BioMistral-7B-0.1-PubMed-V2")
EnsembleModels("Mistral-7B-v0.1", "BioMistral-7B-0.1-PubMed-V2")
print()

print("Mistral-7B-v0.1", " --- ", "Mistral-7B-v0.1")
EnsembleModels("Mistral-7B-v0.1", "Mistral-7B-v0.1")
print()

print("Mistral-7B-Instruct-v0.1", " --- ", "Mistral-7B-v0.1")
EnsembleModels("Mistral-7B-Instruct-v0.1", "Mistral-7B-v0.1")
print()

print("meditron-7b", " --- ", "BioMistral-7B-0.1-PubMed-V2")
EnsembleModels("meditron-7b", "BioMistral-7B-0.1-PubMed-V2")
print()
