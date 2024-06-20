# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BioInstructQA"""

import os
import json
import random
from urllib.request import urlopen

import datasets

from dataclasses import dataclass

random.seed(42)

_DESCRIPTION = """\
Large Language Models (LLMs) have demonstrated remarkable versatility 
in recent years, offering potential applications across specialized 
domains such as healthcare and medicine. Despite the availability of 
various open-source LLMs tailored for health contexts, adapting 
general-purpose LLMs to the medical domain presents significant
challenges. In this paper, we introduce BioMistral, an open-source
LLM tailored for the biomedical domain, utilizing Mistral as its 
foundation model and further pre-trained on PubMed Central. We conduct 
a comprehensive evaluation of BioMistral on a benchmark comprising 10 
established medical question-answering (QA) tasks in English. We also 
explore lightweight models obtained through quantization and model 
merging approaches. Our results demonstrate BioMistral's superior 
performance compared to existing open-source medical models and its 
competitive edge against proprietary counterparts. Finally, to address 
the limited availability of data beyond English and to assess the multilingual 
generalization of medical LLMs, we automatically translated and evaluated this
benchmark into 7 other languages. This marks the first large-scale
multilingual evaluation of LLMs in the medical domain. Datasets, 
multilingual evaluation benchmarks, scripts, and all the models obtained 
during our experiments are freely released.
"""

_HOMEPAGE = "https://huggingface.co/BioMistral"

_LICENSE = "Apache License 2.0"

_URL = "https://huggingface.co/datasets/BioMistral/BioInstructQA/resolve/main/data.zip"

_CITATION = """\
@misc{labrak2024biomistral,
      title={BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains}, 
      author={Yanis Labrak and Adrien Bazoge and Emmanuel Morin and Pierre-Antoine Gourraud and Mickael Rouvier and Richard Dufour},
      year={2024},
      eprint={2402.10373},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

# Few-shot examples should contains : [{"context": "ddd", "question": "ddd", "answer_choices": [{"letter": "A", "text": "ddd"}, {"letter": "B", "text": "ddd"}], "correct_answer": "ddd"}]
def getPrompt(ctx, qst, answ_chs, crt_answ, few_shot_examples=None):
    
    instruction = "The following are multiple choice questions (with answers) about medical knowledge. \n "

    def parseElement(context, question, answer_choices, correct_answer_letter):

        answer_choices = " \n ".join([f"({a['letter'].upper()}) {a['text']}" for a in answer_choices])

        if context != None:
            context = f"{context} \n "
        else:
            context = ""

        return "{{context}}**Question:** {{question}} \n {{answer_choices}} \n **Answer:**({{correct_answer_letter}}" \
        .replace("{{context}}", context) \
        .replace("{{question}}", question) \
        .replace("{{answer_choices}}", answer_choices) \
        .replace("{{correct_answer_letter}}", correct_answer_letter)

    question_answer = parseElement(ctx, qst, answ_chs, crt_answ)

    if few_shot_examples == None:
        prompt = instruction + question_answer
    else:
        
        few_shot_elements = []
        
        for fe in few_shot_examples:
            # print(fe)
            fse = parseElement(fe["context"], fe["question"], [{"letter": o, "text": fe["options"][o]} for o in fe["options"]], fe["correct_answer_letter"])
            # print(fse)
            few_shot_elements.append(fse)
        
        prompt = instruction + " \n ".join(few_shot_elements) + " \n " + question_answer
    
    return prompt

# Few-shot examples should contains : [{"context": "ddd", "question": "ddd", "answer_choices": [{"letter": "A", "text": "ddd"}, {"letter": "B", "text": "ddd"}], "correct_answer": "ddd"}]
def getPromptChat(ctx, qst, answ_chs, crt_answ, few_shot_examples=None):
    
    instruction = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers multiple choice questions about medical knowledge."
        }
    ]

    def parseElement(context, question, answer_choices, correct_answer_letter):

        answer_choices = " ".join([f"”{a['letter'].upper()}”: ”{a['text']}”" for a in answer_choices])

        if context != None:
            context = f"{context} \n "
        else:
            context = ""

        return [
            {
                "role": "user",
                "content": "{{context}}**Question:** {{question}} {{answer_choices}}" \
                .replace("{{context}}", context) \
                .replace("{{question}}", question) \
                .replace("{{answer_choices}}", answer_choices)
            },
            {
                "role": "assistant",
                "content": "**Answer:**({{correct_answer_letter}}" \
                .replace("{{correct_answer_letter}}", correct_answer_letter)
            }
        ]

    question_answer = parseElement(ctx, qst, answ_chs, crt_answ)

    if few_shot_examples == None:
        prompt = instruction + question_answer
    else:
        prompt = instruction + [parseElement(fe["context"], fe["question"], [{"letter": o, "text": fe["options"][o]} for o in fe["options"]], fe["correct_answer_letter"]) for fe in few_shot_examples] + question_answer
    
    return prompt

@dataclass
class CustomConfig(datasets.BuilderConfig):
    name: str = None
    version: datasets.Version = None
    description: str = None
    schema: str = None
    subset_id: str = None

class BioInstructQA(datasets.GeneratorBasedBuilder):
    """BioInstructQA"""

    VERSION = datasets.Version("1.0.6")

    MMLU_configs = [
        CustomConfig(
            name="MMLU_" + subject,
            version=datasets.Version("1.0.6"),
            description=f"Source schema in the raw MMLU format.",
            schema="MMLU_" + subject,
            subset_id="MMLU_" + subject,
        ) for subject in ["clinical_knowledge","medical_genetics","anatomy","professional_medicine","college_biology","college_medicine"]        
    ]

    BUILDER_CONFIGS = [
        CustomConfig(
            name="MedMCQA",
            version=VERSION,
            description="Source schema in the raw MedMCQA format.",
            schema="MedMCQA",
            subset_id="MedMCQA",
        ),
        CustomConfig(
            name="MedQA-5_options",
            version=VERSION,
            description="Source schema in the raw MedQA-5_options format.",
            schema="MedQA-5_options",
            subset_id="MedQA-5_options",
        ),
        CustomConfig(
            name="PubMedQA",
            version=VERSION,
            description="Source schema in the raw PubMedQA format.",
            schema="PubMedQA",
            subset_id="PubMedQA",
        ),
        CustomConfig(
            name="MedQA",
            version=VERSION,
            description="Source schema in the raw MedQA format.",
            schema="MedQA",
            subset_id="MedQA",
        ),
    ] + MMLU_configs

    def _info(self):

        features = datasets.Features(
            {
                "identifier": datasets.Value("string"),
                "corpus_name": datasets.Value("string"),
                "task_type": datasets.Value("string"),
                "classes": [datasets.Value("string")],

                "prompt_no_answer": datasets.Value("string"),
                "prompt": datasets.Value("string"),

                "prompt_fewshot[1]": datasets.Value("string"),
                "prompt_fewshot[2]": datasets.Value("string"),
                "prompt_fewshot[3]": datasets.Value("string"),
                
                "prompt_no_answer_fewshot[1]": datasets.Value("string"),
                "prompt_no_answer_fewshot[2]": datasets.Value("string"),
                "prompt_no_answer_fewshot[3]": datasets.Value("string"),
            }
        )
        
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        
        data_dir = dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split_name": "train",
                    "filepath": os.path.join(data_dir, "./base/overall_train+prompt.json"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split_name": "validation",
                    "filepath": os.path.join(data_dir, "./base/overall_validation+prompt.json"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split_name": "test",
                    "filepath": os.path.join(data_dir, "./base/overall_test+prompt.json"),
                },
            ),
            datasets.SplitGenerator(
                name="test_french",
                gen_kwargs={
                    "split_name": "test_french",
                    "filepath": os.path.join(data_dir, "./langs_processed/French-full.json"),
                },
            ),
            datasets.SplitGenerator(
                name="test_chinese",
                gen_kwargs={
                    "split_name": "test_chinese",
                    "filepath": os.path.join(data_dir, "./langs_processed/Chinese-full.json"),
                },
            ),
            datasets.SplitGenerator(
                name="test_arabic",
                gen_kwargs={
                    "split_name": "test_arabic",
                    "filepath": os.path.join(data_dir, "./langs_processed/Arabic-full.json"),
                },
            ),
            datasets.SplitGenerator(
                name="test_german",
                gen_kwargs={
                    "split_name": "test_german",
                    "filepath": os.path.join(data_dir, "./langs_processed/German-full.json"),
                },
            ),
            datasets.SplitGenerator(
                name="test_portuguese",
                gen_kwargs={
                    "split_name": "test_portuguese",
                    "filepath": os.path.join(data_dir, "./langs_processed/Portuguese-full.json"),
                },
            ),
            datasets.SplitGenerator(
                name="test_russian",
                gen_kwargs={
                    "split_name": "test_russian",
                    "filepath": os.path.join(data_dir, "./langs_processed/Russian-full.json"),
                },
            ),
            datasets.SplitGenerator(
                name="test_spanish",
                gen_kwargs={
                    "split_name": "test_spanish",
                    "filepath": os.path.join(data_dir, "./langs_processed/Spanish-full.json"),
                },
            ),
        ]

    def _generate_examples(self, split_name, filepath):
        
        f = open(filepath)
        data = json.load(f)
        f.close()

        random.seed(42)
        random.shuffle(data)
        
        random.seed(42)
        random.shuffle(data)
        
        random.seed(42)
        random.shuffle(data)

        key = -1

        for d in data:

            if d["corpus_name"] != self.config.name:
                continue

            key += 1

            ctx = None

            if "test_" in split_name:

                d_question = d["question_translated"]
                d_options = d["options_translated"]
                d_correct_answer_letter = d["correct_answer_letter"]

                if d["corpus_name"] == "PubMedQA" and d["context"] != None:
                    ctx = d["context_translated"]
                
            else:

                d_question = d["question"]
                d_options = d["options"]
                d_correct_answer_letter = d["correct_answer_letter"]

                if d["corpus_name"] == "PubMedQA":
                    ctx = d["context"]
            
            messages = getPromptChat(
                ctx=ctx,
                qst=d_question,
                answ_chs=[{"letter": o, "text": d_options[o]} for o in d_options],
                crt_answ=d_correct_answer_letter,
                few_shot_examples=None
            )

            prompt = getPrompt(
                ctx=ctx,
                qst=d_question,
                answ_chs=[{"letter": o, "text": d_options[o]} for o in d_options],
                crt_answ=d_correct_answer_letter,
                few_shot_examples=None
            )

            prompt_no_answer = getPrompt(
                ctx=ctx,
                qst=d_question,
                answ_chs=[{"letter": o, "text": d_options[o]} for o in d_options],
                crt_answ="",
                few_shot_examples=None
            )

            obj = {
                "identifier": d["identifier"],
                "corpus_name": d["corpus_name"],
                "task_type": d["task_type"],
                "classes": d["classes"],
                
                "prompt": prompt,
                "prompt_no_answer": prompt_no_answer,
            }

            for i in range(1,4):

                obj[f"prompt_fewshot[{i}]"] = getPrompt(
                    ctx=ctx,
                    qst=d_question,
                    answ_chs=[{"letter": o, "text": d_options[o]} for o in d_options],
                    crt_answ=d_correct_answer_letter,
                    few_shot_examples=d[f"few_shot_samples[{i}]"]
                )
    
                obj[f"prompt_no_answer_fewshot[{i}]"] = getPrompt(
                    ctx=ctx,
                    qst=d_question,
                    answ_chs=[{"letter": o, "text": d_options[o]} for o in d_options],
                    crt_answ="",
                    few_shot_examples=d[f"few_shot_samples[{i}]"]
                )

            yield key, obj
