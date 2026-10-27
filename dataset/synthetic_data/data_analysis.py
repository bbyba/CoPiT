# import json
# import re
# from collections import defaultdict
# import pandas as pd
# import os

# PAREN_RE = re.compile(r"\([^()]*\)")

# def strip_parentheses(text: str) -> str:
#     if not text or not isinstance(text, str):
#         return text
#     prev = None
#     cur = text
#     while prev != cur:
#         prev = cur
#         cur = PAREN_RE.sub("", cur)
#     return cur


# # 1) Prompt artifact leakage tokens
# ARTIFACT_PATTERNS = [

#     # English
#     r"\btranslation\s*:\s*",
#     r"\btranslate\s*:\s*",
#     r"\btranslated\s*:\s*",
#     r"\banswer\s*:\s*",
#     r"\boutput\s*:\s*",
#     r"\bresponse\s*:\s*",

#     # Korean
#     r"(?:^|[\"'“‘])\s*한국어\s*번역(?:\s*결과)?\s*:\s*",
#     r"(?:^|[\"'“‘])\s*해석(?:\s*결과)?\s*:\s*",
#     r"(?:^|[\"'“‘])\s*출력(?:\s*결과)?\s*:\s*",
#     r"(?:^|[\"'“‘])\s*번역(?:\s*결과)?\s*:\s*",

#     # Russian
#     r"\bперевод\s*:\s*",
#     r"\bрусский\s*перевод\s*:\s*",
#     r"\bроссийский\s*перевод\s*:\s*",
# ]
# ARTIFACT_RE = re.compile("|".join(ARTIFACT_PATTERNS), flags=re.IGNORECASE)
# ARTIFACT_RE = re.compile("|".join(ARTIFACT_PATTERNS), flags=re.IGNORECASE)

# def has_prompt_artifact(text: str) -> bool:
#     if not text or not isinstance(text, str):
#         return False
#     return bool(ARTIFACT_RE.search(text))


# # 2) Sentence-mode drift
# KOR_Q_ENDINGS = (
#     "까", "까요", "나요", "니", "냐", "습니까", "인가", "인가요",
#     "죠?", "죠", "지?", "지요?", "죠요"
# )
# ENG_Q_HINT_RE = re.compile(
#     r"^\s*(who|what|when|where|why|how|is|are|do|does|did|can|could|would|should|will|have|has|had)\b",
#     flags=re.IGNORECASE
# )

# def is_interrogative(text: str, lang_key: str) -> bool:
#     if not text or not isinstance(text, str):
#         return False
#     t = text.strip()

#     if "?" in t:
#         return True

#     if lang_key == "kor_sent":
#         tail = re.sub(r"[\s\.\!\)\]\}\"\']+$", "", t)
#         for end in KOR_Q_ENDINGS:
#             if tail.endswith(end):
#                 return True
#         return False

#     if lang_key == "eng_sent":
#         return bool(ENG_Q_HINT_RE.search(t))

#     return False


# def sentence_mode_drift(record: dict, lang_key: str, source_key: str = "trad_sent") -> bool:
#     src = record.get(source_key, "")
#     tgt = record.get(lang_key, "")

#     if not src or not tgt or lang_key == source_key:
#         return False

#     src_q = is_interrogative(src, source_key)
#     tgt_q = is_interrogative(tgt, lang_key)

#     return src_q != tgt_q


# # 3) Code-mixing
# def script_flags(s: str):
#     flags = {
#         "hangul": False,
#         "cyrillic": False,
#         "latin": False,
#         "mongolian": False,
#         "hanja": False,
#         "arabic": False,
#         "other_letter": False,
#     }
#     if not s:
#         return flags

#     for ch in s:
#         o = ord(ch)

#         if ch.isspace() or ch.isdigit() or re.match(r"[.,;:!?\"'()\[\]{}<>/\\\-—–_+=*&^%$#@~`|]", ch):
#             continue

#         if 0x1800 <= o <= 0x18AF:
#             flags["mongolian"] = True
#             continue

#         if (0xAC00 <= o <= 0xD7A3) or (0x1100 <= o <= 0x11FF) or (0x3130 <= o <= 0x318F):
#             flags["hangul"] = True
#             continue

#         if (0x0400 <= o <= 0x04FF) or (0x0500 <= o <= 0x052F) or (0x2DE0 <= o <= 0x2DFF) or (0xA640 <= o <= 0xA69F):
#             flags["cyrillic"] = True
#             continue

#         if (0x0041 <= o <= 0x007A) or (0x00C0 <= o <= 0x024F) or (0x1E00 <= o <= 0x1EFF):
#             flags["latin"] = True
#             continue

#         if 0x4E00 <= o <= 0x9FFF:
#             flags["hanja"] = True
#             continue

#         if 0x0600 <= o <= 0x06FF:
#             flags["arabic"] = True
#             continue

#         if ch.isalpha():
#             flags["other_letter"] = True

#     return flags


# EXPECTED_SCRIPTS = {
#     "trad_sent": {"mongolian"},
#     "cyrillic_sent": {"cyrillic"},
#     "eng_sent": {"latin"},
#     "kor_sent": {"hangul", "hanja"},
#     "rus_sent": {"cyrillic"},
# }

# def get_scripts(s: str) -> set:
#     flags = script_flags(s)
#     return {k for k, v in flags.items() if v}

# def has_code_mixing(text: str, lang_key: str, trad_text: str) -> bool:
#     if not text or not isinstance(text, str):
#         return False

#     text_wo_paren = strip_parentheses(text)
#     trad_wo_paren = strip_parentheses(trad_text) if trad_text else ""

#     tgt_scripts = get_scripts(text_wo_paren)
#     trad_scripts = get_scripts(trad_wo_paren)

#     expected = EXPECTED_SCRIPTS.get(lang_key, set())

#     allowed = expected.union(trad_scripts)

#     if "other_letter" in tgt_scripts:
#         return True

#     non_expected = tgt_scripts - allowed
#     return len(non_expected) > 0


# def analyze_error_stats(data):
#     lang_keys = [k for k in EXPECTED_SCRIPTS.keys() if any(k in r and r.get(k) for r in data)]

#     counts = {k: defaultdict(int) for k in lang_keys}
#     totals = defaultdict(int)

#     details_rows = []

#     for rec in data:
#         rid = rec.get("idx", None)

#         for lk in lang_keys:
#             txt = rec.get(lk, "")
#             if not txt:
#                 continue
#             totals[lk] += 1

#             artifact = has_prompt_artifact(txt)
#             mixing = has_code_mixing(txt, lk, rec.get("trad_sent", ""))

#             drift = sentence_mode_drift(rec, lk, source_key="trad_sent")

#             if artifact:
#                 counts[lk]["prompt_artifact_leakage"] += 1
#             if drift:
#                 counts[lk]["sentence_mode_drift"] += 1
#             if mixing:
#                 counts[lk]["code_mixing"] += 1

#             details_rows.append({
#                 "record_idx": rid,
#                 "lang": lk,
#                 "prompt_artifact_leakage": artifact,
#                 "sentence_mode_drift": drift,
#                 "code_mixing": mixing,
#                 "text": txt,
#             })

#     rows = []
#     for lk in lang_keys:
#         n = totals[lk]
#         pa = counts[lk]["prompt_artifact_leakage"]
#         sd = counts[lk]["sentence_mode_drift"]
#         cm = counts[lk]["code_mixing"]
#         rows.append({
#             "lang": lk,
#             "n_sentences": n,
#             "prompt_artifact_leakage_cnt": pa,
#             "prompt_artifact_leakage_rate": (pa / n) if n else 0.0,
#             "sentence_mode_drift_cnt": sd,
#             "sentence_mode_drift_rate": (sd / n) if n else 0.0,
#             "code_mixing_cnt": cm,
#             "code_mixing_rate": (cm / n) if n else 0.0,
#         })

#     per_lang_df = pd.DataFrame(rows).sort_values("lang").reset_index(drop=True)
#     per_lang_details_df = pd.DataFrame(details_rows)

#     return per_lang_df, per_lang_details_df


# if __name__ == "__main__":
#     with open("/home/tech/digraphia/.SUBMISSION/dataset/synthetic_data/CoPiT_synthetic_data.json", "r", encoding="utf-8") as f:
#         data = json.load(f)

#     per_lang_df, per_lang_details_df = analyze_error_stats(data)

#     # print("=== Per-language summary ===")
#     # print(per_lang_df.to_string(index=False))

#     # err_samples = per_lang_details_df[
#     #     per_lang_details_df[["prompt_artifact_leakage", "sentence_mode_drift", "code_mixing"]].any(axis=1)
#     # ]
#     # print("\n=== Error samples (first 30) ===")
#     # print(err_samples.head(30)[
#     #     ["record_idx", "lang", "prompt_artifact_leakage", "sentence_mode_drift", "code_mixing", "text"]
#     # ].to_string(index=False))

#     print("\n" + "="*70)
#     print("TRANSLATION ERROR ANALYSIS SUMMARY")
#     print("="*70)

#     for _, row in per_lang_df.iterrows():
#         lang = row["lang"]

#         print(f"\n[{lang}]")
#         print("-"*50)

#         print(f"Total sentences            : {row['n_sentences']}")
#         print(f"Prompt artifact leakage    : {row['prompt_artifact_leakage_cnt']} ({row['prompt_artifact_leakage_rate']*100:.2f}%)")
#         print(f"Sentence-mode drift        : {row['sentence_mode_drift_cnt']} ({row['sentence_mode_drift_rate']*100:.2f}%)")
#         print(f"Code mixing                : {row['code_mixing_cnt']} ({row['code_mixing_rate']*100:.2f}%)")


#     OUTPUT_DIR = "/home/tech/digraphia/.SUBMISSION/dataset/synthetic_data/errors"
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     for lang in per_lang_details_df["lang"].unique():
#         lang_errors = per_lang_details_df[
#             (per_lang_details_df["lang"] == lang) &
#             (
#                 per_lang_details_df["prompt_artifact_leakage"] |
#                 per_lang_details_df["sentence_mode_drift"] |
#                 per_lang_details_df["code_mixing"]
#             )
#         ]

#         if lang_errors.empty:
#             continue

#         out_path = os.path.join(OUTPUT_DIR, f"{lang}_errors.jsonl")

#         with open(out_path, "w", encoding="utf-8") as f:
#             for _, row in lang_errors.iterrows():
#                 rec = {
#                     "idx": row["record_idx"],
#                     "lang": row["lang"],
#                     "errors": {
#                         "prompt_artifact_leakage": bool(row["prompt_artifact_leakage"]),
#                         "sentence_mode_drift": bool(row["sentence_mode_drift"]),
#                         "code_mixing": bool(row["code_mixing"]),
#                     },
#                     "text": row["text"],
#                 }
#                 f.write(json.dumps(rec, ensure_ascii=False) + "\n")

#         print(f"[Saved] {lang}: {len(lang_errors)} error samples → {out_path}")

import json

total = 8034

prompt_artifact_leakage = 0
sentence_mode_drift = 0
code_mixing = 0

with open("/home/tech/digraphia/.SUBMISSION/dataset/synthetic_data/errors/rus_sent_errors.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        errors = data["errors"]

        if errors["prompt_artifact_leakage"]:
            prompt_artifact_leakage += 1
        if errors["sentence_mode_drift"]:
            sentence_mode_drift += 1
        if errors["code_mixing"]:
            code_mixing += 1

print(f"Prompt artifact leakage : {prompt_artifact_leakage} ({prompt_artifact_leakage/total*100:.2f}%)")
print(f"Sentence mode drift     : {sentence_mode_drift} ({sentence_mode_drift/total*100:.2f}%)")
print(f"Code mixing             : {code_mixing} ({code_mixing/total*100:.2f}%)")