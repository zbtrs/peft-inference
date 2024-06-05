from PeftManager import PeftManager,PeftConfig,PeftTask
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
from torch.utils.data import DataLoader
from transformers import default_data_collator
from datasets import load_dataset
from tqdm import tqdm
import os

peft_config = PeftConfig(
    model_name_or_path="bigscience/mt0-large",
    peft_type=TaskType.SEQ_2_SEQ_LM,
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

def preprocess_function(task, examples):
    inputs = examples[task.text_column]
    targets = examples[task.label_column]
    model_inputs = task.tokenizer(inputs, max_length=task.max_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = task.tokenizer(targets, max_length=3, padding="max_length", truncation=True, return_tensors="pt")
    labels = labels["input_ids"]
    labels[labels == task.tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs

def prepare_data_function(task):
    dataset = load_dataset("financial_phrasebank", "sentences_allagree")
    dataset = dataset["train"].train_test_split(test_size=0.1)
    dataset["validation"] = dataset["test"]
    del dataset["test"]

    classes = dataset["train"].features["label"].names
    dataset = dataset.map(
        lambda x: {"text_label": [classes[label] for label in x["label"]]},
        batched=True,
        num_proc=1,
    )

    processed_datasets = dataset.map(
        lambda x: preprocess_function(task, x),
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=task.batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=task.batch_size, pin_memory=True)
    train_iterator = iter(train_dataloader)

    return train_dataset, eval_dataset, train_dataloader, eval_dataloader, train_iterator, dataset

def setup_optimizer_function(task):
    optimizer = torch.optim.AdamW(task.model.parameters(), lr=task.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(task.train_dataloader) * task.num_epochs),
    )
    return optimizer, lr_scheduler

peft_manager = PeftManager()

task1 = PeftTask(
    peft_config=peft_config,
    preprocess_function=preprocess_function,
    prepare_data_function=prepare_data_function,
    setup_optimizer_function=setup_optimizer_function,
    text_column="sentence",
    label_column="text_label",
    max_length=128,
    lr=1e-3,
    batch_size=8,
    num_epochs=1
)

task1.prepare_data()
task1.setup_optimizer_and_scheduler()

task_state = {
    "task": task1,
    "state": None
}

peft_manager.add_task(task_state)

while True:
    peft_manager.run_next_step()

