#!/bin/bash

NBR_RUNS=3

declare -a MODELS=("Project44/BioMedGPT-LM-7B-SFT-PubMedQA-En-8bit-3ep","PharMolix/BioMedGPT-LM-7B")

declare -a CORPUS=('PubMedQA')

for CORPUS_NAME in ${CORPUS[@]}; do
    for MODEL_NAME in ${MODELS[@]}; do

        IFS=',' read peftmodel basemodel <<< "${MODEL_NAME}"

        for ((i=1; i <= $NBR_RUNS; i++)); do
            eval "python PredictFewShot-SFT-Avg-Thread.py --peft_name='$peftmodel' --base_model_name='$basemodel' --corpus='$CORPUS_NAME' --few_shot_run=$i"
        done
    done
done
