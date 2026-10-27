import yaml
import torch
from tqdm import tqdm


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_greedy_batch(model, tokenizer, seg_full_batch, all_prompts, batch_size, desc):
    real_prompts = [p for p in all_prompts if p is not None]

    outputs = []
    for start in range(0, len(real_prompts), batch_size):
        prompts = real_prompts[start:start + batch_size]
        texts = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
        input_ids = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device) 
        input_len = input_ids.input_ids.shape[-1]  

        if "token_type_ids" in input_ids:
            del input_ids["token_type_ids"]

        with torch.no_grad():
            out = model.generate(
                **input_ids,
                max_new_tokens=1024,
                do_sample=False,
                return_dict_in_generate=True,
            )

        gen = out.sequences[:, input_len:]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        outputs.extend([d.strip() for d in decoded])

    result = []
    out_idx = 0

    for seg, prompt in zip(seg_full_batch, all_prompts):
        if prompt is None:
            result.append(seg)
        else:
            result.append(outputs[out_idx])
            out_idx += 1

    return result


def generate_greedy_batch_tqdm(model, tokenizer, seg_full_batch, all_prompts, batch_size, desc):
    real_prompts = [p for p in all_prompts if p is not None]

    outputs = []
    for start in tqdm(range(0, len(real_prompts), batch_size), desc=desc, total=(len(real_prompts) + batch_size - 1) // batch_size):
        prompts = real_prompts[start:start + batch_size]
        texts = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
        input_ids = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)  
        input_len = input_ids.input_ids.shape[-1] 

        if "token_type_ids" in input_ids:
            del input_ids["token_type_ids"]

        with torch.no_grad():
            out = model.generate(
                **input_ids,
                max_new_tokens=1024,
                do_sample=False,
                return_dict_in_generate=True,
            )

        gen = out.sequences[:, input_len:]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        outputs.extend([d.strip() for d in decoded])

    result = []
    out_idx = 0

    for seg, prompt in zip(seg_full_batch, all_prompts):
        if prompt is None:
            result.append(seg)
        else:
            result.append(outputs[out_idx])
            out_idx += 1

    return result