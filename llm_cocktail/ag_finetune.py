from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
    Trainer, DataCollatorForLanguageModeling
)

label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

dataset = load_dataset("ag_news")

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
model.gradient_checkpointing_enable()
model.config.pad_token_id = tokenizer.pad_token_id

def preprocess(example):
    prompt = f"Input: {example['text']}\nLabel:"
    label_text = f" {label_map[example['label']]}"

    full_text = prompt + label_text
    tokenized = tokenizer(full_text, truncation=True, padding="max_length", max_length=128)

    prompt_len = len(tokenizer(prompt, truncation=True, max_length=128)["input_ids"])
    labels = [-100] * prompt_len + tokenized["input_ids"][prompt_len:]
    tokenized["labels"] = labels

    return tokenized

tokenized_dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

training_args = TrainingArguments(
    output_dir="./phi2-agnews-gen",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=2,
    save_total_limit=2,
    num_train_epochs=3,
    evaluation_strategy="no",
    save_strategy="steps",
    learning_rate=2e-5,
    logging_steps=10,
    report_to="wandb"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train(resume_from_checkpoint=True)
