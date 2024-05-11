# LLM Experimentation: PII Instruction Tuning Dataset
The code is a Python script that creates a personal identifiable information dataset based on several BERT PII datasets. We then train a Llama 3 8B model on the PII dataset and evaluate it's performance.
## Language Model Text Generation and Evaluation

This notebook contains a Python script for text generation and evaluation using various transformer-based language models. It leverages the Hugging Face Transformers library for model loading, tokenization, and generation tasks.

### Prerequisites

- Python 3.6+
- PyTorch
- Hugging Face Transformers
- NLTK
- Rouge Score
- BERT Score
- Datasets
- tqdm
- matplotlib

Install the required packages using:

```bash
pip install torch transformers nltk rouge-score bert-score datasets tqdm matplotlib
```

### Code Overview

The script provides functionality for:

1. **Text Generation**: Generating text completions from language models.
2. **Text Evaluation**: Evaluating generated text against reference outputs using BLEU, ROUGE-L, and BERTScore metrics.
3. **Model Loading and Usage**: Loading pre-trained language models (GPT-2, LLM models) and leveraging them for text generation.
4. **Dataset Loading**: Loading datasets for training and testing text generation models.

### Usage

1. **Import Libraries**:
   
   Start by importing the necessary libraries and modules:

   ```python
   import torch.nn.functional as F
   from transformers import AutoTokenizer, AutoModelForCausalLM
   from transformers.tokenization_utils_base import BatchEncoding
   import torch
   import logging
   import matplotlib.pyplot as plt
   from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
   from rouge_score import rouge_scorer
   from bert_score import score
   from datasets import load_dataset, load_metric
   from tqdm.auto import tqdm
   import json
   from peft import AutoPeftModelForCausalLM
   ```


## References

- [Hugging Face Transformers Documentation](https://huggingface.co/transformers/)
- [NLTK Documentation](https://www.nltk.org/)
- [Rouge Score Documentation](https://github.com/google-research/google-research/tree/master/rouge)
- [BERT Score Documentation](https://github.com/Tiiiger/bert_score)
- [Datasets Library Documentation](https://huggingface.co/docs/datasets/)
- [tqdm Documentation](https://github.com/tqdm/tqdm)

## Notes

- Ensure that you have compatible GPU resources if using CUDA for model acceleration (`device = 'cuda'`).
- Experiment with different model configurations (layer, temperature, top_k) for text generation.
- Adjust the dataset and model paths/configurations according to your specific use case.

For further details, refer to the inline comments and documentation within the provided Python script.
