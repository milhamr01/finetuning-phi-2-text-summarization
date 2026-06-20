# Evaluation Summary — Phi-2 XSum Text Summarization

## 1. Experiment Identity

- **Repository:** `finetuning-phi-2-text-summarization`
- **Assignment:** UAS Deep Learning — Fine-Tuning HuggingFace Models
- **Task:** Task 3 — Decoder-only LLM Text Summarization
- **Architecture:** Decoder-only Transformer / Causal Language Model
- **Model:** `microsoft/phi-2`
- **Dataset:** XSum (`EdinburghNLP/xsum`)
- **Problem Type:** Abstractive text summarization

## 2. Dataset Summary

The XSum dataset was loaded successfully from HuggingFace. The dataset contains long news documents and short abstractive summaries. The relevant fields are:

```text
document, summary, id
```

The original dataset contains:

| Split | Size |
|---|---:|
| Training | 204,045 |
| Validation | 11,332 |
| Test | 11,334 |

For practical execution in Google Colab, the notebook used the following subset:

| Split | Size |
|---|---:|
| Training | 300 |
| Validation | 50 |
| Test | 50 |

## 3. Preprocessing and Prompt Format

Because Phi-2 is a decoder-only model, the summarization task was converted into an instruction-style causal language modeling task.

The prompt format was:

```text
Summarize the following news article in one concise sentence.

Document:
<document text>

Summary:
```

The reference summary was appended after the prompt during training so the model learned to continue the text with a summary.

## 4. Model and Fine-Tuning Configuration

| Parameter | Value |
|---|---:|
| Base Model | `microsoft/phi-2` |
| Fine-tuning Method | LoRA |
| Quantization | 4-bit NF4 |
| Trainable Parameters | 5,242,880 |
| Total Parameters | 2,784,926,720 |
| Trainable Percentage | 0.1883% |
| Epochs | 1 |
| Learning Rate | 2e-4 |
| Train Batch Size | 1 |
| Evaluation Batch Size | 1 |
| Gradient Accumulation Steps | 8 |
| Weight Decay | 0.01 |
| Max Sequence Length | 512 |
| Max New Tokens | 64 |
| Hardware | Tesla T4 GPU |

## 5. Training Result

| Metric | Value |
|---|---:|
| Training Loss | 2.470495 |
| Validation Loss | 2.287577 |
| Global Training Loss | 2.449979 |
| Runtime | 274.5579 seconds |

The model loaded successfully with 4-bit quantization, and LoRA training completed for one epoch. Only **0.1883%** of the model parameters were trainable, making the experiment feasible in a limited GPU environment.

## 6. Evaluation Metrics

The generated summaries were evaluated using simple ROUGE-style overlap metrics:

- **ROUGE-1 F1:** unigram overlap between generated and reference summaries.
- **ROUGE-2 F1:** bigram overlap between generated and reference summaries.
- **ROUGE-L F1:** longest common subsequence overlap.

## 7. Final Test Metrics

| Metric | Value |
|---|---:|
| ROUGE-1 F1 | 0.172716 |
| ROUGE-2 F1 | 0.050965 |
| ROUGE-L F1 | 0.130178 |

## 8. Qualitative Prediction Analysis

The generated summaries showed mixed quality:

1. Some summaries captured the main topic of the news article.
2. Some summaries were too long or included extra generated text.
3. Some outputs repeated parts of the original prompt or source document.
4. The model occasionally produced summaries that were topic-related but not aligned closely enough with the XSum reference summary.

This behavior is expected because the experiment used only 300 training examples and one training epoch. XSum is also highly abstractive, so ROUGE overlap can be low when the generated wording differs from the reference.

## 9. Inference Demo

The model was tested on a custom short paragraph about a satellite launch for climate monitoring and weather forecasting.

```text
Generated summary:
A new satellite was launched to monitor climate patterns and improve weather forecasting.
```

The output was concise and captured the main point of the custom input.

## 10. Conclusion

The fine-tuned Phi-2 model achieved **ROUGE-1 = 0.172716**, **ROUGE-2 = 0.050965**, and **ROUGE-L = 0.130178** on the XSum test subset. These results are reasonable for a lightweight LoRA fine-tuning setup using only 300 training samples, 50 validation samples, 50 test samples, and one training epoch.

Phi-2 can be adapted for abstractive summarization using decoder-only instruction-style prompting. However, stronger performance would likely require a larger training subset, additional epochs, improved prompt design, better generation settings, and more extensive hyperparameter tuning.
