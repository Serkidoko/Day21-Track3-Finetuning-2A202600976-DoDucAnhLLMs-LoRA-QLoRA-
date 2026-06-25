# Lab 21 - Evaluation Report

**Hoc vien**: Do Duc Anh - 2A202600976  
**Ngay nop**: 2026-06-25  
**Submission option**: B - GitHub + HuggingFace Hub  

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 samples
- **Split**: 180 train / 20 eval, seed = 42
- **Format**: Alpaca-style `instruction/input/output`, mapped to a single `text` field
- **max_seq_length**: 1024, using the T4 notebook cap after token length analysis
- **GPU**: Google Colab Tesla T4, 16 GB VRAM
- **Training config**: 3 epochs, effective batch size = 8, cosine LR, learning rate = 2e-4, warmup ratio = 0.10, optimizer = `adamw_8bit`
- **LoRA target modules**: `["q_proj", "v_proj"]`
- **Estimated training cost**: about 11.27 minutes total; approximately $0.07 at $0.35/hour, or $0 direct cash cost on free Colab
- **GitHub repo**: https://github.com/Serkidoko/Day21-Track3-Finetuning-2A202600976-DoDucAnhLLMs-LoRA-QLoRA-
- **HF adapters**:
  - r=8: https://huggingface.co/DAnh2580/lab21-qwen25-3b-vi-r8
  - r=16: https://huggingface.co/DAnh2580/lab21-qwen25-3b-vi-r16
  - r=64: https://huggingface.co/DAnh2580/lab21-qwen25-3b-vi-r64

## 2. Rank Experiment Results

| Rank | Alpha | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 1,843,200 | 3.59 min | 7.22 GB | 1.5577 | 4.7479 |
| 16 | 32 | 3,686,400 | 4.11 min | 6.62 GB | 1.5161 | 4.5544 |
| 64 | 128 | 14,745,600 | 3.57 min | 8.00 GB | 1.4768 | 4.3790 |

The quantitative CSV records the three trained adapters. The base model was used in the qualitative before/after comparison, but base-model perplexity was not exported into the summary CSV in this run.

## 3. Loss Curve Analysis

The T4 notebook used `eval_strategy="no"` to avoid mid-training evaluation OOM, so the most reliable quantitative comparison is the final eval loss after training each adapter. The final eval loss decreased as rank increased: r=8 reached 1.5577, r=16 reached 1.5161, and r=64 reached 1.4768. This suggests that higher rank gave the adapter more capacity to fit the Vietnamese instruction-following dataset.

I did not observe a clear overfitting signal from the exported metrics, because eval loss did not increase for the higher-rank adapter. However, the improvement from r=16 to r=64 is smaller than the increase in trainable parameters. r=64 uses 4x the trainable parameters of r=16, but perplexity only improves from 4.5544 to 4.3790. This points to diminishing returns even though the final metric is best for r=64.

## 4. Qualitative Comparison

| Prompt | Base model | Fine-tuned r=16 | Comment |
|---|---|---|---|
| Explain machine learning for beginners | Gives a general definition of machine learning and learning from data. | Also gives a beginner-friendly definition and mentions AI, algorithms, and prediction. | Slightly cleaner, but both are acceptable. |
| Write Python code for the nth Fibonacci number | Starts with a recursive/loop explanation but the exported answer is cut before a complete implementation. | Produces a more standard iterative structure with `ValueError`, base cases, and loop variables. | Fine-tuned answer is more useful for coding. |
| List 5 UI/UX design principles | Gives a broad answer about user friendliness and layout, but it is verbose and truncated. | Lists principles such as conversion, responsiveness, simplicity, and compatibility. | Fine-tuned answer is more compact, though still shallow. |
| Summarize LoRA vs QLoRA | Correctly expands LoRA as Low-Rank Adaptation, but the explanation is truncated. | Incorrectly expands LoRA as "Layer-wise Adaptive Regularization Optimization". | Fine-tuning degraded factual accuracy on this prompt. |
| Distinguish prompt engineering, RAG, and fine-tuning | Gives a partial explanation of the three methods. | Gives a somewhat clearer opening, but still remains generic and truncated. | Similar quality; fine-tuning did not fully solve conceptual precision. |

Overall, the fine-tuned r=16 adapter improved formatting and directness for general Vietnamese instruction prompts. The main weakness is factual precision: on the LoRA/QLoRA prompt, the fine-tuned output introduced an incorrect expansion of LoRA. This is a useful failure case because it shows that fine-tuning can improve style and format without guaranteeing factual correctness.

## 5. Conclusion ve Rank Trade-off

For this dataset, r=64 achieved the best final perplexity, but r=16 gives the best practical ROI. Moving from r=8 to r=16 doubles the trainable parameters and improves perplexity from 4.7479 to 4.5544, which is a meaningful gain. Moving from r=16 to r=64 increases trainable parameters from 3.69M to 14.75M, about 4x larger, but perplexity only improves from 4.5544 to 4.3790. That improvement is real, but the return per additional parameter is much smaller. Peak VRAM is also highest for r=64 at about 8.00 GB. If I were deploying this adapter in a constrained environment, I would choose r=16 because it balances quality, memory, and adapter size. I would choose r=64 only if the task requires maximum accuracy and the serving/training budget can tolerate the larger adapter. For a 200-sample instruction dataset, the results show the classic LoRA trade-off: higher rank increases capacity, but after a point the extra capacity gives diminishing returns.

## 6. What I Learned

- LoRA rank controls adapter capacity directly: increasing rank improves eval loss here, but the gain is not linear compared with parameter growth.
- QLoRA makes small-GPU fine-tuning practical because the base model stays in 4-bit while only LoRA parameters are trained.
- Fine-tuning is better at shifting style and response format than adding guaranteed factual knowledge; factual prompts still need careful evaluation or RAG.
