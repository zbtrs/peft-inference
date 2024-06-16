from PeftManager import PeftManager,PeftConfig,PeftTask
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
from torch.utils.data import DataLoader
from transformers import default_data_collator
from datasets import load_dataset
from tqdm import tqdm
import os

class Config:
    def __init__(self,text_column,label_column,model_name_or_path,max_length,batch_size):
        self.text_column = text_column
        self.label_column = label_column
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.max_length = max_length
        self.batch_size = batch_size

peft_config = PeftConfig(
    model_name_or_path="bigscience/mt0-large",
    peft_type=TaskType.SEQ_2_SEQ_LM,
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

def preprocess_function(config: Config, examples):
    inputs = examples[config.text_column]
    targets = examples[config.label_column]
    model_inputs = config.tokenizer(inputs, max_length=config.max_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = config.tokenizer(targets, max_length=3, padding="max_length", truncation=True, return_tensors="pt")
    labels = labels["input_ids"]
    labels[labels == config.tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs

def prepare_data_function(config: Config):
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
        lambda x: preprocess_function(config,x),
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    return train_dataset, eval_dataset, dataset

def setup_optimizer_function(task):
    optimizer = torch.optim.AdamW(task.model.parameters(), lr=task.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(task.train_dataloader) * task.num_epochs),
    )
    return optimizer, lr_scheduler

peft_manager = PeftManager()


config = Config("sentence","text_label","bigscience/mt0-large",128,8)

train_dataset,eval_dataset,dataset = prepare_data_function(config)
print(train_dataset)
optimizer,lr_scheduler = None,None

task1 = PeftTask(
    peft_config=peft_config,
    text_column="sentence",
    label_column="text_label",
    max_length=128,
    lr=1e-3,
    batch_size=8,
    num_epochs=1,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset=dataset
)

task1.post_init()

task_state = {
    "task": task1,
    "state": None
}

peft_manager.add_task(task_state)

while True:
    peft_manager.run_next_step()