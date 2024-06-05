import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
from torch.utils.data import DataLoader
from transformers import default_data_collator
from datasets import load_dataset
from tqdm import tqdm
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PeftConfig:
    def __init__(self, model_name_or_path, peft_type, task_type, inference_mode, r, lora_alpha, lora_dropout):
        self.model_name_or_path = model_name_or_path
        self.peft_type = peft_type
        self.task_type = task_type
        self.inference_mode = inference_mode
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

    def get_config(self):
        return LoraConfig(
            task_type=self.task_type,
            inference_mode=self.inference_mode,
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
        )

class PeftTask:
    def __init__(self, peft_config, text_column, label_column, max_length, lr, batch_size, num_epochs):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model_name_or_path = peft_config.model_name_or_path
        self.peft_config = peft_config

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name_or_path)
        self.model = get_peft_model(self.model, self.peft_config.get_config())
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        self.current_epoch = 0
        self.current_step = 0

        self.train_dataset = None
        self.eval_dataset = None
        self.train_dataloader = None
        self.eval_dataloader = None

        self.optimizer = None
        self.lr_scheduler = None

    def preprocess_function(self, examples):
        inputs = examples[self.text_column]
        targets = examples[self.label_column]
        model_inputs = self.tokenizer(inputs, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        labels = self.tokenizer(targets, max_length=3, padding="max_length", truncation=True, return_tensors="pt")
        labels = labels["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs

    def prepare_data(self):
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
            self.preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

        self.train_dataset = processed_datasets["train"]
        self.eval_dataset = processed_datasets["validation"]

        self.train_dataloader = DataLoader(
            self.train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=self.batch_size, pin_memory=True
        )
        self.eval_dataloader = DataLoader(self.eval_dataset, collate_fn=default_data_collator, batch_size=self.batch_size, pin_memory=True)

    def setup_optimizer_and_scheduler(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(self.train_dataloader) * self.num_epochs),
        )

    def save_state(self):
        return {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'current_epoch': self.current_epoch,
            'current_step': self.current_step
        }

    def load_state(self, state):
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(state['lr_scheduler_state_dict'])
        self.current_epoch = state['current_epoch']
        self.current_step = state['current_step']

    def train_step(self, num_steps=1):
        if num_steps <= 0:
            raise ValueError("Number of steps must be a positive integer.")
        
        self.model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(self.train_dataloader)):
            if step >= num_steps:
                break
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            self.current_step += 1
        return total_loss

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(self.train_dataloader)):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            self.current_step += 1
        self.current_epoch += 1
        return total_loss / len(self.train_dataloader)

    def evaluate(self):
        self.model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(self.eval_dataloader)):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                self.tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
            )
        return eval_loss / len(self.eval_dataloader), eval_preds

class PeftManager:
    def __init__(self):
        self.task_queue = []
        self.current_task = None

    def add_task(self, peft_task):
        self.task_queue.append(peft_task)
        print(f"Task added. Queue length: {len(self.task_queue)}")

    def next_task(self):
        if not self.task_queue:
            raise ValueError("No tasks in the queue.")
        self.current_task = self.task_queue.pop(0)
        print(f"Starting next task. Remaining queue length: {len(self.task_queue)}")
        return self.current_task

    def run_next_step(self):
        if not self.current_task:
            raise ValueError("No current task. Call next_task() first.")
        try:
            loss = self.current_task.train_step(num_steps=1)
            print(f"Step completed. Loss: {loss}")
            return loss
        except Exception as e:
            print(f"Error during training step: {e}")

# 示例使用

# 定义 PEFT 配置
peft_config = PeftConfig(
    model_name_or_path="bigscience/mt0-large",
    peft_type=TaskType.SEQ_2_SEQ_LM,
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

# 初始化训练任务
task1 = PeftTask(
    peft_config=peft_config,
    text_column="sentence",
    label_column="text_label",
    max_length=128,
    lr=1e-3,
    batch_size=8,
    num_epochs=3
)

# 准备数据
task1.prepare_data()

# 设置优化器和调度器
task1.setup_optimizer_and_scheduler()

# 初始化任务管理器并添加任务
manager = PeftManager()
manager.add_task(task1)

# 运行下一个任务
manager.next_task()

# 进行一步训练并打印损失
manager.run_next_step()

# 保存状态以便下次恢复
state = task1.save_state()

# 恢复状态后继续训练
task1.load_state(state)
manager.run_next_step()
