# Fine-tune DeepSeek-R1-Distill-Llama-8B on Medical COT

## Overview
This repository contains code for fine-tuning the **DeepSeek-R1-Distill-Llama-8B** model on medical reasoning tasks using the **Medical O1 Reasoning SFT** dataset. The fine-tuning process leverages **LoRA (Low-Rank Adaptation)** for efficient parameter tuning and **Unsloth** for optimized inference and training.

## Features
- **Utilizes Unsloth** for efficient loading and inference
- **LoRA Fine-Tuning** for memory-efficient adaptation
- **Medical Reasoning Dataset** sourced from [FreedomIntelligence](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
- **Weights & Biases Integration** for experiment tracking
- **Inference Testing** before and after fine-tuning

## Setup
### Prerequisites
Ensure you have Python installed and the required libraries.

```bash
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file and add your Hugging Face and Weights & Biases tokens:

```
HUGGINGFACE_TOKEN=your_huggingface_token
WANDB_TOKEN=your_wandb_token
```

## Running the Code
### 1. Load Required Packages and Authenticate
```python
from dotenv import load_dotenv
import os
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from huggingface_hub import login
from transformers import TrainingArguments
from datasets import load_dataset
import wandb

load_dotenv()

# Authenticate Hugging Face & Weights & Biases
hugging_face_token = os.getenv("HUGGINGFACE_TOKEN")
wnb_token = os.getenv("WANDB_TOKEN")
login(token=hugging_face_token)
wandb.login(key=wnb_token)
```

### 2. Load the DeepSeek R1 Model
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=2048,
    load_in_4bit=True,
    token=hugging_face_token,
)
```

### 3. Test the Model Before Fine-Tuning
```python
question = """A 61-year-old woman with a long history of involuntary urine loss during activities like coughing or \
              sneezing but no leakage at night undergoes a gynecological exam and Q-tip test. Based on these findings, \
              what would cystometry most likely reveal about her residual volume and detrusor contractions?"""

FastLanguageModel.for_inference(model)
inputs = tokenizer([question], return_tensors="pt").to("cuda")
outputs = model.generate(input_ids=inputs.input_ids, max_new_tokens=1200)
response = tokenizer.batch_decode(outputs)
print(response[0])
```

### 4. Prepare the Fine-Tuning Dataset
```python
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train[0:500]", trust_remote_code=True)

# Formatting function
def formatting_prompts_func(examples):
    texts = []
    for input, cot, output in zip(examples["Question"], examples["Complex_CoT"], examples["Response"]):  
        text = f"""
        ### Instruction:
        You are a medical expert. Please answer the following question.
        
        ### Question:
        {input}
        
        ### Response:
        <think>
        {cot}
        </think>
        {output}
        """ + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

dataset_finetune = dataset.map(formatting_prompts_func, batched=True)
```

### 5. Apply LoRA Fine-Tuning
```python
model_lora = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)
```

### 6. Train the Model
```python
trainer = SFTTrainer(
    model=model_lora,
    tokenizer=tokenizer,
    train_dataset=dataset_finetune,
    dataset_text_field="text",
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        max_steps=60,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        optim="adamw_8bit",
        output_dir="outputs",
    ),
)

trainer.train()
wandb.finish()
```

### 7. Run Inference After Fine-Tuning
```python
FastLanguageModel.for_inference(model_lora)
inputs = tokenizer([question], return_tensors="pt").to("cuda")
outputs = model_lora.generate(input_ids=inputs.input_ids, max_new_tokens=1200)
response = tokenizer.batch_decode(outputs)
print(response[0])
```

## Results
The fine-tuned model is expected to generate improved responses for medical reasoning questions compared to the pre-trained version.

## References
- [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B)
- [Medical O1 Reasoning SFT Dataset](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [Unsloth for Efficient Fine-Tuning](https://github.com/unslothai/unsloth)