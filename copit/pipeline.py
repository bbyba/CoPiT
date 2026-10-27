from utils import generate_greedy_batch, generate_greedy_batch_tqdm

# Script Disambiguation
def stage1_vh_batch(batch_size, seg_full_batch, seg_mask_batch, template, model, tokenizer):
    all_prompts = []
    for seg in seg_mask_batch:
        if seg is None:
            all_prompts.append(None)
        else:
            all_prompts.append([
                {"role": "system", "content": template["stage1"]["system"]},
                {"role": "user", "content": template["stage1"]["user"].format(trad_word=seg)},
            ])
    return generate_greedy_batch(model, tokenizer, seg_full_batch, all_prompts, batch_size, desc="Stage 1")


def stage2_latin_batch(batch_size, seg_full_batch, seg_mask_batch, vh_batch, template, model, tokenizer):
    all_prompts = []
    for seg, vh in zip(seg_mask_batch, vh_batch):
        if seg is None:
            all_prompts.append(None)
        else:
            all_prompts.append([
                {"role": "system", "content": template["stage2"]["system"]},
                {"role": "user", "content": template["stage2"]["user"].format(
                    trad_word=seg, vowel_harmony=vh)},
            ])
    return generate_greedy_batch(model, tokenizer, seg_full_batch, all_prompts, batch_size, desc="Stage 2")


def stage3_cyr_batch(batch_size, seg_full_batch, seg_mask_batch, vh_batch, latin_batch, template, model, tokenizer):
    all_prompts = []
    for seg, vh, latin in zip(seg_mask_batch, vh_batch, latin_batch):
        if seg is None:
            all_prompts.append(None)
        else:
            all_prompts.append([
                {"role": "system", "content": template["stage3"]["system"]},
                {"role": "user", "content": template["stage3"]["user"].format(
                    trad_word=seg, vowel_harmony=vh, latin_trans=latin)},
            ])
    return generate_greedy_batch(model, tokenizer, seg_full_batch, all_prompts, batch_size, desc="Stage 3")



# Reflection
def self_reflect_batch(batch_size, cyrillic_sents, template, model, tokenizer):
    all_prompts = []
    for sent in cyrillic_sents:
        all_prompts.append([
            {"role": "system", "content": template["self_reflect"]["system"]},
            {"role": "user", "content": template["self_reflect"]["user"].format(cyrillic_sent=sent)}, 
        ])
    return generate_greedy_batch_tqdm(model, tokenizer, cyrillic_sents, all_prompts, batch_size, desc="Self-Reflection")



# Translation
def to_eng_batch(batch_size, final_sents, template, model, tokenizer):
    all_prompts = []
    for sent in final_sents:
        all_prompts.append([
            {"role": "system", "content": template["toEnglish"]["system"]},
            {"role": "user", "content": template["toEnglish"]["user"].format(cyrillic_sent=sent)}, 
        ])
    return generate_greedy_batch_tqdm(model, tokenizer, final_sents, all_prompts, batch_size, desc="Translation (Eng)")


def to_kor_batch(batch_size, final_sents, template, model, tokenizer):
    all_prompts = []
    for sent in final_sents:
        all_prompts.append([
            {"role": "system", "content": template["toKorean"]["system"]},
            {"role": "user", "content": template["toKorean"]["user"].format(cyrillic_sent=sent)},
        ])
    return generate_greedy_batch_tqdm(model, tokenizer, final_sents, all_prompts, batch_size, desc="Translation (Kor)")


def to_rus_batch(batch_size, final_sents, template, model, tokenizer):
    all_prompts = []
    for sent in final_sents:
        all_prompts.append([
            {"role": "system", "content": template["toRussian"]["system"]},
            {"role": "user", "content": template["toRussian"]["user"].format(cyrillic_sent=sent)},
        ])
    return generate_greedy_batch_tqdm(model, tokenizer, final_sents, all_prompts, batch_size, desc="Translation (Rus)")