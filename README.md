# Fine-Tuning Phi-2 for Text Summarization

This repository was created for the **UAS Deep Learning — Fine-Tuning HuggingFace Models** assignment.

## Project Purpose

The purpose of this repository is to build an end-to-end decoder-only LLM pipeline for abstractive text summarization using HuggingFace. The project fine-tunes **Phi-2** on the **XSum** dataset using parameter-efficient LoRA fine-tuning and 4-bit quantization to fit the experiment into a Google Colab T4 GPU environment.

## Project Overview

This project covers the full deep learning summarization pipeline:

1. XSum dataset loading and inspection
2. Dataset subset creation for practical Colab training
3. Instruction-style prompt formatting for decoder-only summarization
4. Phi-2 tokenizer preparation
5. 4-bit Phi-2 model loading using BitsAndBytes
6. LoRA parameter-efficient fine-tuning
7. Summary generation on held-out XSum examples
8. ROUGE-style evaluation
9. Inference demo using a custom news paragraph

## Assignment Mapping

- **Task:** Task 3 — Phi-2 Text Summarization
- **Architecture Type:** Decoder-only Transformer / Causal Language Model
- **Model Used:** Phi-2 (`microsoft/phi-2`)
- **Dataset:** XSum (`EdinburghNLP/xsum`)
- **Problem Type:** Abstractive Text Summarization
- **Repository Name:** `finetuning-phi-2-text-summarization`

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── 03_phi2_xsum_summarization.ipynb
└── reports/
    └── evaluation_summary.md
```

## Notebook Description

### `notebooks/03_phi2_xsum_summarization.ipynb`

This notebook contains:

- theoretical explanation of decoder-only Transformer summarization,
- XSum dataset loading,
- dataset field inspection,
- subset creation for train, validation, and test,
- instruction-style prompt formatting,
- Phi-2 tokenizer preparation,
- 4-bit quantized model loading,
- LoRA adapter configuration,
- causal language model fine-tuning,
- summary generation on the test subset,
- ROUGE-style evaluation,
- inference demo with a custom news paragraph.

## Dataset Configuration

The XSum dataset loaded successfully with these fields:

```text
document, summary, id
```

The original dataset contains:

| Split | Size |
|---|---:|
| Training | 204,045 |
| Validation | 11,332 |
| Test | 11,334 |

The experiment used a lightweight subset for practical execution in Google Colab with T4 GPU:

| Split | Size |
|---|---:|
| Training | 300 |
| Validation | 50 |
| Test | 50 |

## Training Configuration

| Parameter | Value |
|---|---:|
| Model | `microsoft/phi-2` |
| Fine-tuning Method | LoRA |
| Quantization | 4-bit NF4 |
| Trainable Parameters | 5,242,880 |
| Total Parameters | 2,784,926,720 |
| Trainable Percentage | 0.1883% |
| Epochs | 1 |
| Learning Rate | 2e-4 |
| Train Batch Size | 1 |
| Eval Batch Size | 1 |
| Gradient Accumulation Steps | 8 |
| Weight Decay | 0.01 |
| Max Sequence Length | 512 |
| Max New Tokens | 64 |
| Hardware | Tesla T4 GPU |

## Final Results

### Training Result

| Metric | Value |
|---|---:|
| Training Loss | 2.470495 |
| Validation Loss | 2.287577 |
| Global Training Loss | 2.449979 |
| Runtime | 274.5579 seconds |

### Test Subset ROUGE-Style Metrics

| Metric | Value |
|---|---:|
| ROUGE-1 F1 | 0.172716 |
| ROUGE-2 F1 | 0.050965 |
| ROUGE-L F1 | 0.130178 |

## Result Analysis

The fine-tuned Phi-2 model achieved **ROUGE-1 = 0.172716**, **ROUGE-2 = 0.050965**, and **ROUGE-L = 0.130178** on the XSum test subset. These scores are modest, but they are reasonable for a lightweight experiment using only 300 training examples, 50 validation examples, 50 test examples, one training epoch, and LoRA-based parameter-efficient fine-tuning.

The generated summaries show mixed behavior. Some generated summaries captured the general topic of the article, while some outputs were too long, repeated parts of the prompt, or added extra text. This is expected in a very small fine-tuning setup for an abstractive summarization dataset like XSum.

## Inference Demo Result

The model was tested on a custom short news paragraph:

| Input Topic | Generated Summary |
|---|---|
| Satellite launch for climate monitoring and weather forecasting | A new satellite was launched to monitor climate patterns and improve weather forecasting. |

The inference demo shows that the model can generate a concise summary for a simple custom paragraph.

## How to Run

### Google Colab Recommended

1. Open `notebooks/03_phi2_xsum_summarization.ipynb` in Google Colab.
2. Enable GPU:

```text
Runtime > Change runtime type > T4 GPU
```

3. Run all cells from top to bottom.
4. Download the executed notebook.
5. Clean notebook widget metadata if GitHub displays an `Invalid Notebook` message.
6. Upload the fixed notebook back to this repository.

### Local Environment

```bash
python -m venv venv
```

Activate the environment:

```bash
# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Jupyter Notebook:

```bash
jupyter notebook
```

## Identification

- **Name:** MUHAMMAD ILHAM RAYANDA
- **Class:** Deep Learning
- **NIM:** Fill your NIM here
- **Group:** Fill your group name/member list here

## Conclusion

This repository successfully implements an end-to-end Phi-2 fine-tuning pipeline for XSum abstractive text summarization. The model achieved **ROUGE-1 = 0.172716**, **ROUGE-2 = 0.050965**, and **ROUGE-L = 0.130178** on the test subset. The result demonstrates that Phi-2 can be adapted for summarization using instruction-style prompting and LoRA, although stronger performance would require more training data, longer training, better prompt design, and additional hyperparameter tuning.

## Notes

This repository is prepared as a reproducible academic deep learning project. The code is original/adapted for the assignment and is intended for learning purposes.
