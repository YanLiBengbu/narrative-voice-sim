"""Utterance- and session-level measures.

Measures are grouped into
  * manipulation checks  - pronoun profile (directly targeted by the mode instruction);
  * outcome measures     - sentiment, subjectivity, lexical measures computed on content
                           words with pronouns and character names removed, so that the
                           trivial I -> <Name> substitution cannot drive them;
  * interaction measures - turn-taking balance, addressee responsiveness, repetition.
"""

import math
import os
import re
import sys
from collections import Counter

import numpy as np

FIRST = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
SECOND = {"you", "your", "yours", "yourself", "yourselves"}
THIRD = {"he", "him", "his", "himself", "she", "her", "hers", "herself", "they", "them", "their", "theirs", "themselves"}
NEU_THRESH = 0.05


class Featurizer:
    def __init__(self, use_transformer=True):
        import nltk
        import spacy
        from textblob import TextBlob
        from wordfreq import zipf_frequency

        nltk.data.path.insert(0, os.path.expanduser("~/.nltk_private"))
        from nltk.sentiment import SentimentIntensityAnalyzer

        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        self.TextBlob, self.zipf = TextBlob, zipf_frequency
        self.vader = SentimentIntensityAnalyzer()
        self.clf = None
        if use_transformer:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                name = "distilbert-base-uncased-finetuned-sst-2-english"
                self.tok = AutoTokenizer.from_pretrained(name)
                self.clf = AutoModelForSequenceClassification.from_pretrained(name).eval()
                self.torch = torch
            except Exception as e:  # fall back to TextBlob polarity, as in the original release
                print(f"[metrics] DistilBERT unavailable ({e}); using TextBlob only", file=sys.stderr)

    def distilbert(self, texts):
        if self.clf is None:
            return [float("nan")] * len(texts)
        out = []
        for i in range(0, len(texts), 32):
            x = self.tok(texts[i:i + 32], return_tensors="pt", padding=True, truncation=True, max_length=256)
            with self.torch.no_grad():
                p = self.clf(**x).logits.softmax(-1).numpy()
            out.extend((p[:, 1] - p[:, 0]).tolist())  # signed score in [-1, 1]
        return out

    def utterances(self, rows, names):
        """rows: list of dicts with 'text' and 'speaker'. names: all character names of the scene."""
        name_tok = {w.lower() for n in names for w in re.split(r"[\s\-]+", n) if w and w.lower() not in {"the", "a", "of"}}
        texts = [r["text"] for r in rows]
        db = self.distilbert(texts)
        feats = []
        for r, doc, d in zip(rows, self.nlp.pipe(texts, batch_size=64), db):
            alpha = [t for t in doc if t.is_alpha]
            low = [t.lower_ for t in alpha]
            content = [t.lower_ for t in alpha if not t.is_stop and t.pos_ not in ("PRON", "PROPN")
                       and t.lower_ not in name_tok and t.lower_ not in FIRST | SECOND | THIRD]
            fp, sp, tp = sum(w in FIRST for w in low), sum(w in SECOND for w in low), sum(w in THIRD for w in low)
            spk = {w.lower() for w in re.split(r"[\s\-]+", r.get("speaker", "")) if w}
            blob = self.TextBlob(r["text"])
            pol, subj = blob.sentiment.polarity, blob.sentiment.subjectivity
            z = [self.zipf(w, "en") for w in content]
            feats.append(dict(
                n_words=len(alpha), n_content=len(content),
                fp_pron=fp, sp_pron=sp, tp_pron=tp,
                fp_ratio=fp / max(fp + tp, 1), fp_rate=fp / max(len(alpha), 1), tp_rate=tp / max(len(alpha), 1),
                self_name=sum(w in spk for w in low),
                polarity=pol, intensity=abs(pol), subjectivity=subj,
                vader=self.vader.polarity_scores(r["text"])["compound"],
                distilbert=d,
                sent_class="pos" if pol > NEU_THRESH else ("neg" if pol < -NEU_THRESH else "neu"),
                word_len_all=float(np.mean([len(w) for w in low])) if low else np.nan,
                word_len=float(np.mean([len(w) for w in content])) if content else np.nan,
                zipf=float(np.mean(z)) if z else np.nan,
                rare_rate=float(np.mean([v < 3.0 for v in z])) if z else np.nan,
                question=int("?" in r["text"]),
                content_tokens=content, tokens=low,
            ))
        return feats


def mattr(tokens, window=50):
    if not tokens:
        return np.nan
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    c = Counter(tokens[:window])
    vals = [len(c) / window]
    for i in range(window, len(tokens)):
        c[tokens[i]] += 1
        old = tokens[i - window]
        c[old] -= 1
        if c[old] == 0:
            del c[old]
        vals.append(len(c) / window)
    return float(np.mean(vals))


def _mtld_pass(tokens, thr=0.72):
    factors, types, n = 0.0, set(), 0
    for w in tokens:
        n += 1
        types.add(w)
        if len(types) / n <= thr:
            factors += 1
            types, n = set(), 0
    if n > 0:
        ttr = len(types) / n
        factors += (1 - ttr) / (1 - thr) if ttr < 1 else 0
    return len(tokens) / factors if factors > 0 else np.nan


def mtld(tokens):
    if len(tokens) < 10:
        return np.nan
    return float(np.nanmean([_mtld_pass(tokens), _mtld_pass(tokens[::-1])]))


def distinct_n(token_lists, n=2):
    grams = [tuple(t[i:i + n]) for t in token_lists for i in range(len(t) - n + 1)]
    return len(set(grams)) / len(grams) if grams else np.nan


def session_measures(turns, feats, characters, opening_last=None):
    spk = [t["speaker"] for t in turns]
    k = max(len(characters), 2)
    cnt = Counter(spk)
    probs = np.array([cnt.get(c, 0) for c in characters], dtype=float)
    probs = probs[probs > 0] / max(len(spk), 1)
    balance = float(-(probs * np.log(probs)).sum() / math.log(k)) if len(spk) else np.nan
    prev = [opening_last] + turns[:-1]
    resp = [t["speaker"] in (p.get("speak_to") or []) for p, t in zip(prev, turns) if p]
    succ = [a == b for a, b in zip(spk, spk[1:])]
    content = [w for f in feats for w in f["content_tokens"]]
    alltok = [w for f in feats for w in f["tokens"]]
    out = dict(
        n_turns=len(turns), turn_balance=balance,
        responsiveness=float(np.mean(resp)) if resp else np.nan,
        self_succession=float(np.mean(succ)) if succ else np.nan,
        mattr=mattr(content), mtld=mtld(alltok),
        distinct2=distinct_n([f["tokens"] for f in feats], 2),
        repetition=1 - distinct_n([f["tokens"] for f in feats], 3) if feats else np.nan,
    )
    for flag in ("compliant_first", "regenerated", "director_rewrite", "residual_violation"):
        vals = [t[flag] for t in turns if flag in t]
        out[flag] = float(np.mean(vals)) if vals else np.nan
    return out


# ---------------------------------------------------------------------------
# Rule-based "mechanical transposition" baseline: first-person direct speech is
# converted to third-person reported speech purely by pronoun substitution and a
# reporting frame, without any change of content or style.
# ---------------------------------------------------------------------------
_SUBJ = {"female": "she", "male": "he"}
_OBJ = {"female": "her", "male": "him"}
_POSS = {"female": "her", "male": "his"}
_POSS2 = {"female": "hers", "male": "his"}
_REFL = {"female": "herself", "male": "himself"}


def transpose_to_third(text, speaker, gender="unknown", addressees=None):
    g = gender if gender in ("female", "male") else "male"
    subj, obj, poss, poss2, refl = _SUBJ[g], _OBJ[g], _POSS[g], _POSS2[g], _REFL[g]
    t = text.strip().strip('"“”')
    rules = [
        (r"\bI am\b|\bI'm\b", f"{subj} was"), (r"\bI have\b|\bI've\b", f"{subj} had"), (r"\bI will\b|\bI'll\b", f"{subj} would"),
        (r"\bI do not\b|\bI don't\b", f"{subj} did not"), (r"\bI can\b", f"{subj} could"), (r"\bI\b", subj),
        (r"\bme\b", obj), (r"\bmyself\b", refl), (r"\bmy\b", poss), (r"\bmine\b", poss2),
        (r"\bwe\b", "they"), (r"\bus\b", "them"), (r"\bour\b", "their"), (r"\bours\b", "theirs"), (r"\bourselves\b", "themselves"),
        (r"\byou\b", "them"), (r"\byour\b", "their"), (r"\byours\b", "theirs"), (r"\byourself\b", "themselves"),
    ]
    for pat, rep in rules:
        t = re.sub(pat, lambda m, r=rep: r.capitalize() if m.group(0)[0].isupper() and m.start() == 0 else r, t, flags=re.I)
    to = ", ".join(a for a in (addressees or []) if a) or "the others"
    return f"{speaker} told {to} that {t[0].lower() + t[1:] if t else t}"
