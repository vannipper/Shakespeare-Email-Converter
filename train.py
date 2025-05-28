# Using AutoTrain on Hugging Face

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch

# Model name
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Load full dataset
dataset = load_dataset("ayaan04/english-to-shakespeare")

# output dir
output_dir = "./shakespeare_modelv1"

# Format prompts before splitting
def format_prompt(example):
    instruction = example["instruction"]
    input_text = example["input"]
    output = example["output"]
    full_prompt = f"<|system|>\n{instruction}\n<|user|>\n{input_text}\n<|assistant|>\n{output}"
    return {"text": full_prompt}

dataset = dataset.map(format_prompt)

# Split into 5% train, 95% eval
split_dataset = dataset["train"].train_test_split(test_size=0.95, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Tokenize
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # Fix padding issue

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_train = train_dataset.map(tokenize, batched=True)
tokenized_eval = eval_dataset.map(tokenize, batched=True)

tokenizer.save_pretrained(output_dir)

# Load model

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# Training
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    optim="adamw_torch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
model.save_pretrained(output_dir, safe_serialization=True)
