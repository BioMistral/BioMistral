<p align="center">
<!--   <img src="https://huggingface.co/BioMistral/BioMistral-7B/resolve/main/Image%2BTitle.png?download=true" alt="drawing" width="600"/> -->
  <img src="https://huggingface.co/BioMistral/BioMistral-7B/resolve/main/wordart_blue_m_rectangle.png?download=true" alt="drawing" width="250"/>
</p>

BioMistral 7B is the best performing open-source medical Large Language Model (LLM) in this weight category. 

We've publicly released the weights for BioMistral 7B on Huggingface and our full multilingual benchmark on GitHub.

- 📰 Paper: BioMistral: <a href="https://arxiv.org/abs/2402.10373"> A Collection of Open-Source Pretrained Large Language Models for Medical Domains</a> (Accepted at ACL 2024)

- 📊 Multilingual medical benchmark : <a href="https://huggingface.co/datasets/BioMistral/BioInstructQA">BioMistral/BioInstructQA</a>

- 🤗 BioMistral 7B Chat - A free demonstrator (running on a A10 24GB GPU thanks to Hugging Face grant): <a href="https://huggingface.co/spaces/BioMistral/BioMistral-Chat">BioMistral/BioMistral-Chat</a>

- 👩‍💻 GitHub: <a href="https://github.com/BioMistral/BioMistral">BioMistral/BioMistral</a>

This project is the result of the collaboration between:

<table>
<tbody>
  <tr>
    <td>🏛️ <a href="https://lia.univ-avignon.fr/">LIA - Avignon University (1)</a></td>
    <td>🏛️ <a href="https://www.ls2n.fr/">LS2N - Nantes University (3)</a></td>
  </tr>
  <tr>
    <td>🏥 <a href="https://www.chu-nantes.fr/unite-recherche-2">Nantes University Hospital (2)</a></td>
    <td>🏢 <a href="https://zenidoc.fr/">Zenidoc (4)</a></td>
  </tr>
</tbody>
</table>

**Authors** : Yanis LABRAK (1,4) ; Adrien BAZOGE (2,3) ; Emmanuel MORIN (3) ; Pierre-Antoine GOURAUD (2) ; Mickaël ROUVIER (1) ; Richard DUFOUR (3)

**BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains**

**Abstract:**

Large Language Models (LLMs) have demonstrated remarkable versatility in recent years, offering potential applications across specialized domains such as healthcare and medicine. Despite the availability of various open-source LLMs tailored for health contexts, adapting general-purpose LLMs to the medical domain presents significant challenges.
In this paper, we introduce BioMistral, an open-source LLM tailored for the biomedical domain, utilizing Mistral as its foundation model and further pre-trained on PubMed Central. We conduct a comprehensive evaluation of BioMistral on a benchmark comprising 10 established medical question-answering (QA) tasks in English. We also explore lightweight models obtained through quantization and model merging approaches. Our results demonstrate BioMistral's superior performance compared to existing open-source medical models and its competitive edge against proprietary counterparts. Finally, to address the limited availability of data beyond English and to assess the multilingual generalization of medical LLMs, we automatically translated and evaluated this benchmark into 7 other languages. This marks the first large-scale multilingual evaluation of LLMs in the medical domain. Datasets, multilingual evaluation benchmarks, scripts, and all the models obtained during our experiments are freely released.

**Advisory Notice!** Although BioMistral is intended to encapsulate medical knowledge sourced from high-quality evidence, it hasn't been tailored to effectively, safely, or suitably convey this knowledge within professional parameters for action. We advise refraining from utilizing BioMistral in medical contexts unless it undergoes thorough alignment with specific use cases and undergoes further testing, notably including randomized controlled trials in real-world medical environments. BioMistral 7B may possess inherent risks and biases that have not yet been thoroughly assessed. Additionally, the model's performance has not been evaluated in real-world clinical settings. Consequently, we recommend using BioMistral 7B strictly as a research tool and advise against deploying it in production environments for natural language generation or any professional health and medical purposes.

# 1. BioMistral models

**BioMistral** is a suite of Mistral-based further pre-trained open source models suited for the medical domains and pre-trained using textual data from PubMed Central Open Access (CC0, CC BY, CC BY-SA, and CC BY-ND). All the models are trained using the CNRS (French National Centre for Scientific Research) [Jean Zay](http://www.idris.fr/jean-zay/) French HPC.

|      Model Name     |             Base Model             |      Model Type     | Sequence Length |                          Download                          |
|:-------------------:|:----------------------------------:|:-------------------:|:---------------:|:-----------------------------------------------------:|
|    BioMistral-7B    | [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) | Further Pre-trained |       2048      |    [HuggingFace](https://huggingface.co/BioMistral/BioMistral-7B)    |
|  BioMistral-7B-DARE | [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) |      Merge DARE     |       2048      |  [HuggingFace](https://huggingface.co/BioMistral/BioMistral-7B-DARE) |
|  BioMistral-7B-TIES | [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) |      Merge TIES     |       2048      |  [HuggingFace](https://huggingface.co/BioMistral/BioMistral-7B-TIES) |
| BioMistral-7B-SLERP | [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) |     Merge SLERP     |       2048      | [HuggingFace](https://huggingface.co/BioMistral/BioMistral-7B-SLERP) |

# 2. Quantized Models

|      Base Model     | Method | q_group_size | w_bit | version | VRAM GB |  Time  | Download |
|:-------------------:|:------:|:------------:|:-----:|:-------:|:-------:|:------:|:--------:|
|    BioMistral-7B    | FP16/BF16       |              |   |       |  15.02  |  x1.00 | [HuggingFace](https://huggingface.co/BioMistral/BioMistral-7B)         |
|    BioMistral-7B    |   AWQ  |      128     |   4   |   GEMM  |   4.68  |  x1.41 | [HuggingFace](https://huggingface.co/BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM)         |
|    BioMistral-7B    |   AWQ  |      128     |   4   |   GEMV  |   4.68  | x10.30 | [HuggingFace](https://huggingface.co/BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMV)         |
|    BioMistral-7B    |  BnB.4 |              |   4   |         |   5.03  |  x3.25 | [HuggingFace](blank)         |
|    BioMistral-7B    |  BnB.8 |              |   8   |         |   8.04  |  x4.34 | [HuggingFace](blank)         |
|  BioMistral-7B-DARE |   AWQ  |      128     |   4   |   GEMM  |   4.68  |  x1.41 | [HuggingFace](https://huggingface.co/BioMistral/BioMistral-7B-DARE-AWQ-QGS128-W4-GEMM)         |
|  BioMistral-7B-TIES |   AWQ  |      128     |   4   |   GEMM  |   4.68  |  x1.41 | [HuggingFace](https://huggingface.co/BioMistral/BioMistral-7B-TIES-AWQ-QGS128-W4-GEMM)         |
| BioMistral-7B-SLERP |   AWQ  |      128     |   4   |   GEMM  |   4.68  |  x1.41 | [HuggingFace](https://huggingface.co/BioMistral/BioMistral-7B-SLERP-AWQ-QGS128-W4-GEMM)         |

# 2. Using BioMistral

You can use BioMistral with [Hugging Face's Transformers library](https://github.com/huggingface/transformers) as follow.

Loading the model and tokenizer :

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B")
model = AutoModel.from_pretrained("BioMistral/BioMistral-7B")
```

# 3. Supervised Fine-tuning Benchmark

| | Clinical KG | Medical Genetics | Anatomy | Pro Medicine | College Biology | College Medicine | MedQA | MedQA 5 opts | PubMedQA | MedMCQA | Avg. |
|-------------------------------------------|:---------------------------------------------:|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|------------------|
| **BioMistral 7B** | 59.9 | 64.0 | 56.5 | 60.4 | 59.0 | 54.7 | 50.6 | 42.8 | 77.5 | 48.1 | 57.3 |
| **Mistral 7B Instruct** | **62.9** | 57.0 | 55.6 | 59.4 | 62.5 | <u>57.2</u> | 42.0 | 40.9 | 75.7 | 46.1 | 55.9 |
| | | | | | | | | | | | |
| **BioMistral 7B Ensemble** | <u>62.8</u> | 62.7 | <u>57.5</u> | **63.5** | 64.3 | 55.7 | 50.6 | 43.6 | 77.5 | **48.8** | 58.7 |
| **BioMistral 7B DARE** | 62.3 | **67.0** | 55.8 | 61.4 | **66.9** | **58.0** | **51.1** | **45.2** | <u>77.7</u> | <u>48.7</u> | **59.4** |
| **BioMistral 7B TIES** | 60.1 | <u>65.0</u> | **58.5** | 60.5 | 60.4 | 56.5 | 49.5 | 43.2 | 77.5 | 48.1 | 57.9 |
| **BioMistral 7B SLERP** | 62.5 | 64.7 | 55.8 | <u>62.7</u> | <u>64.8</u> | 56.3 | <u>50.8</u> | <u>44.3</u> | **77.8** | 48.6 | <u>58.8</u> |
| | | | | | | | | | | | |
| **MedAlpaca 7B** | 53.1 | 58.0 | 54.1 | 58.8 | 58.1 | 48.6 | 40.1 | 33.7 | 73.6 | 37.0 | 51.5 |
| **PMC-LLaMA 7B** | 24.5 | 27.7 | 35.3 | 17.4 | 30.3 | 23.3 | 25.5 | 20.2 | 72.9 | 26.6 | 30.4 |
| **MediTron-7B** | 41.6 | 50.3 | 46.4 | 27.9 | 44.4 | 30.8 | 41.6 | 28.1 | 74.9 | 41.3 | 42.7 |
| **BioMedGPT-LM-7B** | 51.4 | 52.0 | 49.4 | 53.3 | 50.7 | 49.1 | 42.5 | 33.9 | 76.8 | 37.6 | 49.7 |
| | | | | | | | | | | | |
| **GPT-3.5 Turbo 1106*** | 74.71 | 74.00 | 65.92 | 72.79 | 72.91 | 64.73 | 57.71 | 50.82 | 72.66 | 53.79 | 66.0 |

Supervised Fine-Tuning (SFT) performance of BioMistral 7B models compared to baselines, measured by accuracy (↑) and averaged across 3 random seeds of 3-shot. DARE, TIES, and SLERP are model merging strategies that combine BioMistral 7B and Mistral 7B Instruct. Best model in bold, and second-best underlined. *GPT-3.5 Turbo performances are reported from the 3-shot results without SFT.

# Citation BibTeX

Arxiv : [https://arxiv.org/abs/2402.10373](https://arxiv.org/abs/2402.10373)

```bibtex
@misc{labrak2024biomistral,
      title={BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains}, 
      author={Yanis Labrak and Adrien Bazoge and Emmanuel Morin and Pierre-Antoine Gourraud and Mickael Rouvier and Richard Dufour},
      year={2024},
      eprint={2402.10373},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

**CAUTION!** Both direct and downstream users need to be informed about the risks, biases, and constraints inherent in the model. While the model can produce natural language text, our exploration of its capabilities and limitations is just beginning. In fields such as medicine, comprehending these limitations is crucial. Hence, we strongly advise against deploying this model for natural language generation in production or for professional tasks in the realm of health and medicine.



