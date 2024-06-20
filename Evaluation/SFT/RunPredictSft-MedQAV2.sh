#!/bin/bash

NBR_RUNS=3

declare -a MODELS=("BioMistral/SFT-BioMistral-7B-MedQA","Project44/BioMistral-7B-0.1-PubMed-V2")

declare -a CORPUS=('MedQA')

for CORPUS_NAME in ${CORPUS[@]}; do
    for MODEL_NAME in ${MODELS[@]}; do

        IFS=',' read peftmodel basemodel <<< "${MODEL_NAME}"

        for ((i=1; i <= $NBR_RUNS; i++)); do
            eval "python PredictFewShot-SFT-Avg-Thread.py --peft_name='$peftmodel' --base_model_name='$basemodel' --corpus='$CORPUS_NAME' --few_shot_run=$i"
        done
    done
done
