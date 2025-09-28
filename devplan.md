# Dev Plan: Fine-tuning ProGen3 with LoRA for TasR Proteins

This document outlines the plan to fine-tune the ProGen3 protein language model on a custom dataset of TasR protein sequences using Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA).

## 1. Environment Setup

- **Install necessary libraries**: The `pyproject.toml` already includes `transformers`, `torch`, and `accelerate`. We will need to add `peft` for LoRA and `trl` for supervised fine-tuning.

```bash
pip install peft trl
```

## 2. Data Preparation

- **Load Data**: Load the TasR protein sequences from a file (e.g., CSV or FASTA). The sequences will need to be formatted into a Hugging Face `Dataset` object. Each entry in the dataset should contain the full protein sequence text.
- **Tokenization**: Use the specific ProGen3 tokenizer located at `src/progen3/tokenizer.py` to process the sequences. This is crucial for ensuring the input data is compatible with the model.

## 3. Model Loading and PEFT Configuration

- **Load Model**: Load the pre-trained `ProGen3ForCausalLM` model using the classes available in `src/progen3/modeling.py`. The model should be loaded in half-precision (`bfloat16`) to save memory.
- **Configure LoRA**:
    - Create a `LoraConfig` from the `peft` library.
    - This configuration will specify the LoRA parameters, such as:
        - `r` (the rank of the update matrices)
        - `lora_alpha` (the scaling factor)
        - `target_modules` (the modules of the transformer to apply LoRA to, e.g., `q_proj`, `v_proj`)
        - `lora_dropout`
        - `task_type` (set to `CAUSAL_LM`)
    - Use `get_peft_model` to wrap the base ProGen3 model with the LoRA configuration.

## 4. Fine-Tuning

- **Setup Trainer**: Use the `SFTTrainer` from the `trl` library, which is designed for supervised fine-tuning of language models.
- **Training Arguments**: Configure `TrainingArguments` from the `transformers` library to specify:
    - `output_dir`
    - `per_device_train_batch_size`
    - `gradient_accumulation_steps`
    - `learning_rate`
    - `num_train_epochs`
    - `logging_steps`
    - `save_steps`
- **Start Training**: Instantiate the `SFTTrainer` with the model, dataset, tokenizer, and training arguments, and call the `train()` method.

## 5. Save and Evaluate

- **Save Adapter**: After training is complete, save the trained LoRA adapter weights using `model.save_pretrained("path/to/adapter")`. This will save only the lightweight adapter, not the full model.
- **Inference**:
    - To use the fine-tuned model, load the base ProGen3 model again.
    - Load the LoRA adapter from the saved directory using `PeftModel.from_pretrained(base_model, "path/to/adapter")`.
    - Use the merged model for generating new TasR protein sequences.

## 6. Implementation

A Python script will be created to encapsulate these steps. The script will be structured as follows:

```python
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from progen3.modeling import ProGen3ForCausalLM
from progen3.tokenizer import ProGen3Tokenizer

# 1. Load and prepare data
# ... (load sequences into a list)
# dataset = Dataset.from_dict({"text": sequences})

# 2. Load tokenizer and model
# tokenizer = ProGen3Tokenizer.from_pretrained(...)
# model = ProGen3ForCausalLM.from_pretrained(...)

# 3. Configure PEFT
# lora_config = LoraConfig(...)
# model = get_peft_model(model, lora_config)

# 4. Set up and run trainer
# training_args = TrainingArguments(...)
# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     dataset_text_field="text",
#     max_seq_length=1024, # Or other appropriate length
#     tokenizer=tokenizer,
#     args=training_args,
# )
# trainer.train()

# 5. Save model
# trainer.save_model("./tasr-finetuned-adapter")
```
