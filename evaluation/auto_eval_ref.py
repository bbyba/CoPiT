"""
Automatic Evaluation for Reference-based Data
=> BLEU-3/4 / chrF/chrF++ / COMET
"""
import json
import argparse
from sacrebleu.metrics import BLEU
from sacrebleu import corpus_chrf
from comet import download_model, load_from_checkpoint


# load data
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# src, hyp, ref alignment
def align_by_idx(task, test_data, pred_data): 
    pred_dict = {ex["idx"]: ex for ex in pred_data} 

    srcs, hyps, refs = [], [], []

    for ex in test_data: 
        idx = ex["idx"]
        if idx not in pred_dict:
            raise ValueError(f"Missing prediction for idx={idx}")

        # source sentence (for COMET)
        srcs.append(ex["cyrillic_sent"].strip())

        # hypothesis & reference sentence
        if task == "eng":
            hyp = pred_dict[idx]["eng_sent"]
            ref = ex["eng_sent"]
        elif task == "kor":
            hyp = pred_dict[idx]["kor_sent"]
            ref = ex["kor_sent"]
        elif task == "rus":
            hyp = pred_dict[idx]["rus_sent"]
            ref = ex["rus_sent"]
        else:
            raise ValueError(f"Unknown task: {task}")

        hyps.append(hyp.strip())
        refs.append(ref.strip())

    return srcs, hyps, refs


# BLEU
def compute_bleu(task, hyps, refs):
    if task in ["eng", "rus"]:
        bleu3 = BLEU(max_ngram_order=3)
        bleu4 = BLEU(max_ngram_order=4)
    elif task == "kor":
        bleu3 = BLEU(max_ngram_order=3, tokenize="ko-mecab")
        bleu4 = BLEU(max_ngram_order=4, tokenize="ko-mecab")

    result3 = bleu3.corpus_score(hyps, [refs]).score
    result4 = bleu4.corpus_score(hyps, [refs]).score
    return result3, result4


# chrF / chrF++
def compute_chrf(hyps, refs, word_order=0):
    return corpus_chrf(hyps, [refs], word_order=word_order).score


# COMET
def load_comet_model():
    model_path = download_model("Unbabel/wmt22-comet-da")
    return load_from_checkpoint(model_path)


def compute_comet(model, srcs, hyps, refs):
    samples = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    outputs = model.predict(samples, batch_size=8, gpus=1, progress_bar=True)
    return sum(outputs.scores) / len(outputs.scores)


# main
def main():
    test_data = load_json("/home/tech/digraphia/.SUBMISSION/dataset/test_data/reference_based_test.json")
    pred_data = load_json(f"/home/tech/digraphia/outputs/fewshot/fewshot_copit/ministral14B_copitfs.json")

    langs = ["eng", "kor", "rus"]
    results = {}

    comet_model = load_comet_model()

    for lang in langs:
        srcs, hyps, refs = align_by_idx(lang, test_data, pred_data)

        bleu3, bleu4 = compute_bleu(lang, hyps, refs)
        chrf = compute_chrf(hyps, refs)
        chrf_pp = compute_chrf(hyps, refs, word_order=2)
        comet = compute_comet(comet_model, srcs, hyps, refs)

        results[lang] = {
            "BLEU-3": bleu3,
            "BLEU-4": bleu4,
            "chrF": chrf,
            "chrF++": chrf_pp,
            "COMET": comet
        }

    print("=" * 60)
    for lang, scores in results.items():
        print(f"\n[{lang.upper()}]")
        print(f"BLEU-3  : {scores['BLEU-3']:.2f}")
        print(f"BLEU-4  : {scores['BLEU-4']:.2f}")
        print(f"chrF    : {scores['chrF']:.2f}")
        print(f"chrF++  : {scores['chrF++']:.2f}")
        print(f"COMET   : {scores['COMET']:.3f}")


if __name__ == "__main__":
    main()

