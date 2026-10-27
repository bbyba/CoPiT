from tqdm import tqdm
import os
import argparse
import pandas as pd
import json
import torch
from transformers import logging
logging.set_verbosity_error()

from utils import load_yaml
from segmentor import segmentor
from reconstructor import reconstructor
from model_loader import load_base_model, load_trained_backbone, build_multi_adapter_model, set_stage_adapter
from pipeline import stage1_vh_batch, stage2_latin_batch, stage3_cyr_batch, self_reflect_batch, to_eng_batch, to_kor_batch, to_rus_batch


def main(args):
    if args.variant == "prompting":
        base, tokenizer = load_base_model(backbone_key=args.backbone)
    elif args.variant == "trained":
        base_model, tokenizer, lora_dirs = load_trained_backbone(backbone_key=args.backbone)
        model = build_multi_adapter_model(base_model, lora_dirs)

    pipe_template = load_yaml("prompts/pipeline_rule.yaml")
    reflect_template = load_yaml("prompts/self_reflect.yaml")
    translate_template = load_yaml("prompts/translation.yaml")
    test_dir = "ref_test_data.csv"  

    test_df = pd.read_csv(test_dir, encoding="utf-8")
    print(f"Test data: {test_dir}")


    # collect all segments across sentences
    all_seg_full = []  # flattened full segments
    all_seg_mask = []  # flattened masked segments
    sent_to_seg_indices = []

    for _, row in test_df.iterrows():
        trad_sent = row["t_sent"]
        seg_full, seg_mask = segmentor(trad_sent)

        start = len(all_seg_full)
        end = start + len(seg_full)

        all_seg_full.extend(seg_full)
        all_seg_mask.extend(seg_mask)
        sent_to_seg_indices.append(list(range(start, end)))

    total_segs = len(all_seg_full)
    print(f"Total sentences: {len(test_df)}")
    print(f"Total segments:  {total_segs}")


    # run batch stages
    batch_size = args.batch_size

    vh_all = [None] * total_segs
    latin_all = [None] * total_segs
    cyr_all = [None] * total_segs

    for start in tqdm(range(0, total_segs, batch_size), desc="Batched Stage 1–3"):
        end = min(start + batch_size, total_segs)

        seg_full_batch = all_seg_full[start:end]
        seg_mask_batch = all_seg_mask[start:end]

        if args.variant == "prompting":
            vh_batch = stage1_vh_batch(batch_size, seg_full_batch, seg_mask_batch, pipe_template, base, tokenizer)
            vh_all[start:end] = vh_batch

            latin_batch = stage2_latin_batch(batch_size, seg_full_batch, seg_mask_batch, vh_batch, pipe_template, base, tokenizer)
            latin_all[start:end] = latin_batch

            cyr_batch = stage3_cyr_batch(batch_size, seg_full_batch, seg_mask_batch, vh_batch, latin_batch, pipe_template, base, tokenizer)
            cyr_all[start:end] = cyr_batch

        elif args.variant == "trained":
            set_stage_adapter(model, "stage1")
            vh_batch = stage1_vh_batch(batch_size, seg_full_batch, seg_mask_batch, pipe_template, model, tokenizer)
            vh_all[start:end] = vh_batch

            set_stage_adapter(model, "stage2")
            latin_batch = stage2_latin_batch(batch_size, seg_full_batch, seg_mask_batch, vh_batch, pipe_template, model, tokenizer)
            latin_all[start:end] = latin_batch

            set_stage_adapter(model, "stage3")
            cyr_batch = stage3_cyr_batch(batch_size, seg_full_batch, seg_mask_batch, vh_batch, latin_batch, pipe_template, model, tokenizer)
            cyr_all[start:end] = cyr_batch


    # reconstruction per sentence
    print("Reconstructing sentences")
    cyrillic_sents_list = []
    results_struct = []

    for idx, row in test_df.iterrows():
        seg_indices = sent_to_seg_indices[idx]
        cyr_segments = [c if c is not None else "" for c in [cyr_all[i] for i in seg_indices]]
        cyrillic_sent = reconstructor(cyr_segments).lower()
        cyrillic_sents_list.append(cyrillic_sent)
        
        results_struct.append({
            "idx": idx + 1,
            "trad_sent": row["t_sent"],
            "trad_segs_full": [all_seg_full[i] for i in seg_indices],
            "vh_segs": [vh_all[i] for i in seg_indices],
            "latin_segs": [latin_all[i] for i in seg_indices],
            "cyrillic_sent": cyrillic_sent
        })


    new_batch_size = 8

    # self-reflect
    print("Running Batch Self-Reflect")
    if args.variant == "prompting":
        final_sents_list = self_reflect_batch(new_batch_size, cyrillic_sents_list, reflect_template, base, tokenizer)

    elif args.variant == "trained":
        set_stage_adapter(model, "reflect")
        final_sents_list = self_reflect_batch(new_batch_size, cyrillic_sents_list, reflect_template, model, tokenizer)


    pretrans_json = f"{args.name}_pretrans.json"

    pre_results = []
    for i, res in enumerate(results_struct):
        tmp = dict(res)
        tmp["final_sent"] = final_sents_list[i]
        pre_results.append(tmp)

    with open(pretrans_json, "w", encoding="utf-8") as f:
        json.dump(pre_results, f, ensure_ascii=False, indent=2)
    print(f"Saved before-translation JSON to: {pretrans_json}")


    # change model for translation
    if args.variant == "trained":
        print("Dropping LoRA model before translation...")
        del model
        del base_model
        torch.cuda.empty_cache()

        print("Reloading clean base model for translation...")
        base, tokenizer = load_base_model(backbone_key=args.backbone)


    # batch translation
    print("Running Batch Translations")
    jsonl_output = f"{args.name}.jsonl"
    json_output  = f"{args.name}.json"

    open(jsonl_output, "w").close()
    with open(jsonl_output, "a", encoding="utf-8") as f:
        for start in range(0, len(final_sents_list), new_batch_size):
            end = min(start + new_batch_size, len(final_sents_list))
            batch_final = final_sents_list[start:end]

            eng_preds = to_eng_batch(new_batch_size, batch_final, translate_template, base, tokenizer)
            kor_preds = to_kor_batch(new_batch_size, batch_final, translate_template, base, tokenizer)
            rus_preds = to_rus_batch(new_batch_size, batch_final, translate_template, base, tokenizer)

            for i in range(end - start):
                idx = start + i
                res = results_struct[idx]

                res["final_sent"] = final_sents_list[idx]
                res["eng_sent"] = eng_preds[i]
                res["kor_sent"] = kor_preds[i]
                res["rus_sent"] = rus_preds[i]

                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                f.flush()

    # save json
    final_results = []
    with open(jsonl_output, "r", encoding="utf-8") as f:
        for line in f:
            final_results.append(json.loads(line))

    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    print(f"Saved final JSON to: {json_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, choices=["prompting", "trained"], default="trained")
    parser.add_argument("--backbone", type=str, required=True, choices=["qwen4B", "qwen30B", "ministral3B", "ministral14B"])
    parser.add_argument("--name", type=str, required=True, default="output")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    main(args)