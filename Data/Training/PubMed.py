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

"""PubMed"""

import os
import json

import datasets

from dataclasses import dataclass

_DESCRIPTION = """\
ddd
"""

_HOMEPAGE = "ddd"

_LICENSE = "Apache License 2.0"

_URL = "https://huggingface.co/datasets/Project44/PubMed/resolve/main/corpus_pubmed.zip"

_CITATION = """\
ddd
"""

@dataclass
class CustomConfig(datasets.BuilderConfig):
    name: str = None
    version: datasets.Version = None
    description: str = None
    schema: str = None
    subset_id: str = None

class PubMed(datasets.GeneratorBasedBuilder):
    """PubMed"""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        CustomConfig(
            name="default",
            version=VERSION,
            description="Source schema in the paragraph format.",
            schema="default",
            subset_id="default",
        ),
    ]

    def _info(self):

        features = datasets.Features(
            {
                "text": datasets.Value("string"),
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
                    "filepath": os.path.join(data_dir, "corpus_pubmed.json"),
                },
            ),
        ]

    def _generate_examples(self, filepath):

        f_in = open(filepath)
        data = json.load(f_in)
        f_in.close()
        
        words_limit = 1000
    
        key = -1

        for d in data:

            line = d["text"]

            words = line.split(" ")

            if len(words) <= words_limit:
                
                key += 1
                
                yield key, {
                    "text": line,
                }

            else:

                chunks = []
                for chunk_idx in range(0, len(words), words_limit):
                    chunks.append(words[chunk_idx:chunk_idx + words_limit])
    
                for chunk in chunks:
                    key += 1
    
                    yield key, {
                        "text": " ".join(chunk),
                    }
