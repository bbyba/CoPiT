import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers import TrainingArguments as HFTrainingArguments
from typing import Optional
import dataclasses
from datasets import Dataset
from trl import SFTTrainer
from peft import LoraConfig
import pandas as pd
import yaml
import wandb
import argparse

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclasses.dataclass
class MyTrainingArguments(HFTrainingArguments):
    push_to_hub_token: Optional[str] = None

parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=int)
parser.add_argument("--name", type=str)
args = parser.parse_args()

# folder name
run_name = f"{args.name}"

# wandb config
config_defaults = {
    # experiment meta
    "stage": args.stage,
    "name": run_name,

    # core training hyperparams
    "batch_size": 4,
    "grad_accum": 4,
    "lr": 0.0001,
    "epochs": 3,
    "warmup_ratio": 0.1,
    "lr_scheduler": "cosine",

    # lora params
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],

    # precision & optimizer
    "use_bf16": True,
    "optim": "paged_adamw_32bit",

    # logging & saving
    "logging_steps": 20,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
}

config = wandb.init(
    project=f"stage{args.stage}_finetuning",
    name=run_name,
    config=config_defaults
).config

# save configs
wandb_config_yaml = os.path.join(wandb.run.dir, f"{run_name}_config.yaml")
with open(wandb_config_yaml, "w", encoding="utf-8") as f: 
    yaml.dump(config_defaults, f, allow_unicode=True, sort_keys=False) 
print("Uploaded configs to wandb Files.")


# model & tokenizer
model_name = "train_model_name"

model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token == None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model.config.use_cache = False
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()


# load data & template
train_df = pd.read_csv("dataset/train_data/t2c_words_train.csv")
val_df = pd.read_csv("dataset/train_data/t2c_words_val.csv")

# rule prompt
with open("prompts/pipeline_rule.yaml", "r", encoding="utf-8") as f:
    template = yaml.safe_load(f)

# for stage 1 data
def stage1_messages(df, template):
    out = []
    system_prompt = template["stage1"]["system"]
    user_template = template["stage1"]["user"]

    for _, row in df.iterrows():
        trad_word = row["Word_Traditional"]
        vowel_harmony = row["VHarmony"]

        user_prompt = user_template.format(trad_word=trad_word)

        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": vowel_harmony},
        ]
        out.append(message)
    return out

# for stage 2 data
def stage2_messages(df, template):
    out = []
    system_prompt = template["stage2"]["system"]
    user_template = template["stage2"]["user"]

    for _, row in df.iterrows():
        trad_word = row["Word_Traditional"]
        vowel_harmony = row["VHarmony"]
        latin_trans = row["Phonetic_Assembly_Ground_Truth"]

        user_prompt = user_template.format(trad_word=trad_word, vowel_harmony=vowel_harmony)

        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": latin_trans},
        ]
        out.append(message)
    return out

# for stage 3 data
def stage3_messages(df, template):
    out = []
    system_prompt = template["stage3"]["system"]
    user_template = template["stage3"]["user"]

    for _, row in df.iterrows():
        trad_word = row["Word_Traditional"]
        vowel_harmony = row["VHarmony"]
        latin_trans = row["Phonetic_Assembly_Ground_Truth"]
        cyrillic_word = row["Word_Cyrillic"]

        user_prompt = user_template.format(trad_word=trad_word, latin_trans=latin_trans, vowel_harmony=vowel_harmony)
        
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": cyrillic_word},
        ]
        out.append(message)
    return out


# construct data
MAX_LEN = 4096
def make_example_from_messages(messages):
    # full chat text
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # label = assistant answer
    assistant_text = messages[-1]["content"]
    answer_pos = full_text.rfind(assistant_text)
 
    # context = everything before the assistant answer
    ctx_text = full_text[:answer_pos]

    # tokenize
    ctx_enc = tokenizer(
        ctx_text,
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
        return_tensors=None,
    )
    full_enc = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
        return_tensors=None,
    )

    ctx_input_ids = ctx_enc["input_ids"]
    ctx_len = len(ctx_input_ids)

    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]

    # build labels (ignore context, supervise only answer)
    labels = [-100] * len(input_ids)
    for i in range(ctx_len, len(input_ids)):
        labels[i] = input_ids[i]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def build_dataset(msgs):
    examples = [make_example_from_messages(m) for m in msgs]
    return Dataset.from_list(examples)

# dynamic padding
def dynamic_padding_collator(examples):
    max_len = max(len(ex["input_ids"]) for ex in examples)

    def pad(seq, pad_id):
        if len(seq) < max_len:
            seq = seq + [pad_id] * (max_len - len(seq))
        return torch.tensor(seq, dtype=torch.long)

    input_ids = torch.stack([pad(ex["input_ids"], tokenizer.pad_token_id) for ex in examples])
    attention_mask = torch.stack([pad(ex["attention_mask"], 0) for ex in examples])
    labels = torch.stack([pad(ex["labels"], -100) for ex in examples])

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# LoRA config
peft_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    lora_dropout=config.lora_dropout,
    target_modules=config.target_modules,
    bias="none",
    task_type="CAUSAL_LM",
)


# training args
training_args = MyTrainingArguments(
    output_dir=f"lora_adapter/{run_name}",
    per_device_train_batch_size=config.batch_size,
    gradient_accumulation_steps=config.grad_accum,
    num_train_epochs=config.epochs,
    learning_rate=config.lr,
    warmup_ratio=config.warmup_ratio,
    lr_scheduler_type=config.lr_scheduler,
    optim=config.optim,

    # precision
    bf16=config.use_bf16,
    fp16=not config.use_bf16,

    # logging / eval / saving
    logging_steps=config.logging_steps,
    eval_strategy=config.eval_strategy,
    save_strategy=config.save_strategy,
    save_total_limit=3,
    load_best_model_at_end=True,

    # performance
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,

    # misc
    seed=42,
    report_to="wandb",
    run_name = run_name
)

# stages
if args.stage == 1:
    train_msgs = stage1_messages(train_df, template)
    val_msgs   = stage1_messages(val_df, template)
    print("Stage1 training loaded.")
elif args.stage == 2:
    train_msgs = stage2_messages(train_df, template)
    val_msgs   = stage2_messages(val_df, template)
    print("Stage2 training loaded.")
elif args.stage == 3:
    train_msgs = stage3_messages(train_df, template)
    val_msgs   = stage3_messages(val_df, template)
    print("Stage3 training loaded.")

train_data = build_dataset(train_msgs)
val_data = build_dataset(val_msgs)

# trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    processing_class=tokenizer,
    args=training_args,
    data_collator=dynamic_padding_collator,
    peft_config=peft_config,
)

# trainable parameters
def print_trainable_parameters(m):
    trainable, total = 0, 0
    for _, p in m.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"Trainable parameters: {trainable} ({100 * trainable / total:.2f}% of total)")
print_trainable_parameters(trainer.model)
trainer.train()

# save best model !!
trainer.save_model(f"{run_name}")