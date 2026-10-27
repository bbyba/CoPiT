def normalize_punct(seg: str):
    mapping = {
        "·": ".",     # name separator
        "᠂": ",",     # mongolian comma
        "᠃": ".",     # mongolian sentence dot
    }
    for k, v in mapping.items():
        seg = seg.replace(k, v)
    return seg


# punctuation requiring no space before
PUNCT_NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", "…", "—", "–", ")", "〕", "]", "}", "》", "〉", "）", "】"}

# punctuation requiring no space after
PUNCT_NO_SPACE_AFTER = {"(", "〔", "[", "{", "《", "〈", "（", "【"}


def reconstructor(cyr_segments):
    sentence = ""

    for seg in cyr_segments:
        if not seg:
            continue

        seg = normalize_punct(seg.strip())

        # closing punctuation: stick to previous word
        if seg in PUNCT_NO_SPACE_BEFORE:
            sentence += seg
            continue

        # opening punctuation: no space after
        if seg in PUNCT_NO_SPACE_AFTER:
            if sentence == "":
                sentence = seg  # start of sentence
            else:
                sentence += " " + seg
            continue

        # normal word
        if sentence == "":
            sentence = seg
        else:
            sentence += " " + seg

    return sentence.strip()


