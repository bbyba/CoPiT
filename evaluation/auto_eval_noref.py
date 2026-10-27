"""
Automatic Evaluation for Reference-free Data
=> COMETKiwi
"""
import json
from comet import download_model, load_from_checkpoint
import torch

torch.set_float32_matmul_precision("high")

# load data
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# src, hyp, ref alignment 
def align_by_idx(task, test_data, pred_data):
    pred_dict = {ex["idx"] + 1: ex for ex in pred_data} 

    srcs, hyps = [], []

    for ex in test_data: 
        idx = ex["idx"]
        if idx not in pred_dict:
            raise ValueError(f"Missing prediction for idx={idx}")

        # source sentence (for COMET)
        srcs.append(ex["cyrillic_sent"].strip())

        # hypothesis & reference sentence
        if task == "eng":
            hyp = pred_dict[idx]["eng_sent"]
        elif task == "kor":
            hyp = pred_dict[idx]["kor_sent"]
        elif task == "rus":
            hyp = pred_dict[idx]["rus_sent"]
        else:
            raise ValueError(f"Unknown task: {task}")

        hyps.append(hyp.strip())

    return srcs, hyps


# COMETKiwi
def compute_cometqe(srcs, hyps, model):
    samples = [{"src": s, "mt": h} for s, h in zip(srcs, hyps)]
    outputs = model.predict(samples, batch_size=16, gpus=1, progress_bar=True)
    return sum(outputs.scores) / len(outputs.scores)


def main():
    test_data = load_json("test_data.json")
    pred_data = load_json(f"pred_data.json")

    results = {}

    model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")
    model = load_from_checkpoint(model_path)

    for lang in ["eng", "kor", "rus"]:
        srcs, hyps = align_by_idx(lang, test_data, pred_data)
        score = compute_cometqe(srcs, hyps, model)

        results[lang] = score
        print(f"COMETKiwi ({lang}) : {score:.3f}")

    print("\n===== Summary =====")
    for lang, score in results.items():
        print(f"{lang.upper():>3}: {score:.3f}")

if __name__ == "__main__":
    main()

