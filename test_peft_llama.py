import os
import torch
import gc
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,AutoTokenizer
from transformers import TrainingArguments,pipeline
from peft import LoraConfig, PeftModel, get_peft_config
from trl import SFTTrainer
import warnings

warnings.filterwarnings("ignore")

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

device_map = "auto"
df = pd.read_csv("./medquad.csv")

data = Dataset.from_pandas(pd.DataFrame(data=df))
model_name = "/data/aigc/llama2"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
    )
model.config.pretraining_tp = 1
torch.cuda.empty_cache()

LORA_ALPHA = 16
LORA_DROPOUT = 0.2
LORA_R = 64

peft_config = LoraConfig(
        lora_alpha= LORA_ALPHA,
        lora_dropout= LORA_DROPOUT,
        r= LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
    )

LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
BATCH_SIZE = 16
WEIGHT_DECAY = 0.001
MAX_GRAD_NORM = 0.3
gradient_accumulation_steps = 16
STEPS = 1
OPTIM = "adam"
MAX_STEPS = 10

OUTPUT_DIR = "./results"

training_args = TrainingArguments(
    output_dir= OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps= gradient_accumulation_steps,
    learning_rate= LEARNING_RATE,
    logging_steps= STEPS,
    num_train_epochs= NUM_EPOCHS,
    max_steps= MAX_STEPS,
)

torch.cuda.empty_cache()
trainer = SFTTrainer(
    model=model,
    train_dataset=data,
    peft_config=peft_config,
    dataset_text_field="question",
    max_seq_length=500,
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()
