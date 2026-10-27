# Case / postposition suffixes
case_suffixes = {
    "ᠶ᠋ᠢᠨ", "ᠢᠢᠨ", # genitive -yin (vowel-final)
    "ᠦ᠋ᠨ", "ᠤᠨ",  # genitive -ün (cons-final, except n)
    "ᠤ᠋",   "ᠦ",   # genitive -u (n-final)
    "ᠳ᠋ᠤ", "ᠳ᠋ᠤᠷ",  # dative-locative
    "ᠲᠦ", "ᠳᠤᠷ", "ᠲᠤ",   # dative-locative (masc cons)
    "ᠶ᠋ᠢ",          # accusative (vowel-final)
    "ᠢ᠋",            # accusative (cons-final)
    "ᠠ᠋ᠴᠠ",        # ablative
    "ᠪᠠᠷ",          # instrumental (vowel-final)
    "ᠢ᠋ᠶᠡᠷ", "ᠢᠶᠡᠷ",     # instrumental (cons-final)
    "ᠲᠠᠢ",          # comitative
    "ᠤᠷᠤᠭᠤ",        # directive
}

# Plural suffixes
plural_suffixes = {
    "ᠨᠠᠷ",      # -nar / -ner
    "ᠨᠤᠭᠤᠳ",    # -nuγud / -nügüd
    "ᠤ᠋ᠳ",      # -ud / -üd
}

# Simple possessive for people (mini, čini, etc.)
person_poss_suffixes = {
    "ᠮᠢᠨᠢ", "ᠮᠠᠨᠢ",   # mini / mani
    "ᠴᠢᠨᠢ", "ᠲᠠᠨᠢ",   # čini / tani
    "ᠨᠢ", "ᠠᠨᠤ", "ᠢᠨᠦ", # anu / inü
}

# Simple reflexive-possessive suffixes (separate word)
simple_refl_suffixes = {
    "ᠪᠠᠨ",      # -ban / -ben (vowel-final words)
    "ᠢ᠋ᠶᠡᠨ",    # -iyan / -iyen (cons-final words)
}

# Compound reflexive+case forms (we treat them as single suffix segments)
compound_refl_suffixes = {
    "ᠳ᠋ᠡᠭᠠᠨ", "ᠳ᠋ᠡᢉᠡᠨ", "ᠳᠠᠭᠠᠨ", "ᠳᠡᠭᠡᠨ",  # -daγan / -degen / -taγan / -tegen
    "ᠶᠤᠭᠠᠨ", "ᠶᠤᠭᠡᠨ",                        # -yuγan / -yügen
    "ᠠᠴᠠᠭᠠᠨ", "ᠡᠴᠡᠭᠡᠨ",                    # -ačaγan / -ečegen
    "ᠲᠠᠢᠭᠠᠨ", "ᠲᠡᠢᠭᠡᠨ",                    # -tayiγan / -teyigen
}

all_suffixes = case_suffixes | plural_suffixes | person_poss_suffixes | simple_refl_suffixes | compound_refl_suffixes

PUNCTUATION = {"᠃", "᠂", ".", ",", "?", "!", ";", ":", "…", "—", "–", "∙"}
MIDDLE_DOT = "·"
BRACKETS = {"(", ")", "[", "]", "{", "}", "〈", "〉", "《", "》", "〔", "〕", "（", "）", "【", "】"}

TRAILING_PUNCT = PUNCTUATION | BRACKETS 
ALL_PUNCT = PUNCTUATION | BRACKETS | {MIDDLE_DOT}

def _is_mongolian_letter(ch):
    return "\u1800" <= ch <= "\u18AF"

def _has_letter(chs):
    return any(c.isalpha() or _is_mongolian_letter(c) for c in chs)

def _is_abbreviation_or_initial(tok):
    if MIDDLE_DOT in tok or "." in tok:
        core = tok.strip(".")
        return _has_letter(core)
    return False

    
def segmentor(sentence):
    tokens = [t for t in sentence.split() if t.strip()]
    seg_full, seg_mask = [], []

    for tok in tokens:

        # leading brackets
        while len(tok) > 1 and tok[0] in BRACKETS:
            seg_full.append(tok[0])
            seg_mask.append(None)
            tok = tok[1:]

        # trailing punctuation
        while len(tok) > 1 and tok[-1] in TRAILING_PUNCT:
            core = tok[:-1]
            punct = tok[-1]

            seg_full.append(core)
            seg_mask.append(core)

            seg_full.append(punct)
            seg_mask.append(None)

            tok = None
            break

        if tok is None:
            continue

        # abbreviation or initial
        if tok.endswith(".") and len(tok) > 1 and _is_abbreviation_or_initial(tok[:-1]):
            core = tok[:-1]
            seg_full.append(core)
            seg_full.append(".")
            seg_mask.append(core)
            seg_mask.append(None)
            continue

        if _is_abbreviation_or_initial(tok):
            seg_full.append(tok)
            seg_mask.append(tok)
            continue

        # standalone punctuation or bracket
        if tok in ALL_PUNCT:
            seg_full.append(tok)
            seg_mask.append(None)
            continue

        # suffix merging
        if tok in all_suffixes:
            if seg_full and seg_full[-1] not in ALL_PUNCT:
                seg_full[-1] = seg_full[-1] + "-" + tok
                if seg_mask[-1] is not None:
                    seg_mask[-1] = seg_mask[-1] + "-" + tok
            continue

        # normal word
        seg_full.append(tok)
        seg_mask.append(tok)

    return seg_full, seg_mask
