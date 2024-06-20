#!/bin/bash

NBR_RUNS=3

declare -a MODELS=("Project44/BioMedGPT-LM-7B-SFT-MedQA-5opt-En-8bit-3ep","PharMolix/BioMedGPT-LM-7B" "Project44/BioMistral-7B-Instruct-ties-SFT-MedQA-5opt-En-8bit-3ep","Project44/BioMistral-7B-Instruct-ties" "Project44/BioMistral-7B-Instruct-dare-SFT-MedQA-5opt-En-8bit-3ep","Project44/BioMistral-7B-Instruct-dare" "Project44/BioMistral-7B-Instruct-slerp-SFT-MedQA-5opt-En-8bit-3ep","Project44/BioMistral-7B-Instruct-slerp")
# declare -a MODELS=("Project44/Mistral-Instruct-SFT-MedQA-5opt-En-8bit-3ep","mistralai/Mistral-7B-Instruct-v0.1" "Project44/MedAlpaca-SFT-MedQA-5opt-En-8bit-3ep","medalpaca/medalpaca-7b" "Project44/MediTron-SFT-MedQA-5opt-En-8bit-3ep","epfl-llm/meditron-7b" "Project44/BioMistral-Instruct-SFT-MedQA-5opt-En-8bit-3ep","Project44/BioMistral-7B-0.1-PubMed-V2" "Project44/PMC-LLAMA-SFT-MedQA-5opt-En-8bit-3ep","chaoyi-wu/PMC_LLAMA_7B")

declare -a CORPUS=('MedQA-5_options' 'MMLU')

for CORPUS_NAME in ${CORPUS[@]}; do
    for MODEL_NAME in ${MODELS[@]}; do

        IFS=',' read peftmodel basemodel <<< "${MODEL_NAME}"
        # echo "${peftmodel}" and "${basemodel}"

        for ((i=1; i <= $NBR_RUNS; i++)); do
            eval "python PredictFewShot-SFT-Avg-Thread.py --peft_name='$peftmodel' --base_model_name='$basemodel' --corpus='$CORPUS_NAME' --few_shot_run=$i"
        done
    done
done
