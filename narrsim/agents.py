"""Director Agent (scene initialisation, narrative-mode enforcement) and Actor Agents
(role-conditioned turn generation), plus a single-prompt baseline generator."""

import difflib
import json
import re

from . import prompts
from .jsonfix import parse_json

FIRST_PRON = re.compile(r"\b(i|me|my|mine|myself)\b", re.I)
SPEECH_PRON = re.compile(r"\b(i|i'm|i'll|i've|i'd|me|my|mine|myself|we|us|our|ours|you|your|yours)\b", re.I)
THIRD_REF = re.compile(r"\b(he|she|him|her|his|hers|they|them|their)\b", re.I)
REPORT_VERB = r"(said|says|told|tells|replied|asked|declared|insisted|explained|expressed|answered|cried|added|urged|begged|remarked|exclaimed|murmured|whispered)"
QUOTE = re.compile(r"[\"“”]")
GENERIC = {"all", "everyone", "everybody", "self", "himself", "herself", "narrator", "the narrator", "nobody", "others", "the others"}


def _fmt_history(turns):
    if not turns:
        return "(the scene begins)"
    return "\n".join(f"{t['speaker']} (to {', '.join(t.get('speak_to') or []) or 'all'}): {t['text']}" for t in turns)


class CallStats:
    """Counts generations, JSON repairs, retries and mode corrections (reported in the paper)."""

    def __init__(self):
        self.calls = self.repaired = self.parse_fail = self.mode_violations = self.mode_fixed = 0

    def as_dict(self):
        return dict(calls=self.calls, repaired=self.repaired, parse_fail=self.parse_fail,
                    mode_violations=self.mode_violations, mode_fixed=self.mode_fixed)


def call_json(llm, prompt, seed, stats, max_attempts=3, max_tokens=None, temperature=None):
    """Generate and parse JSON with a fixed retry budget; retries use derived seeds."""
    last = None
    for a in range(max_attempts):
        raw = llm.generate(prompt, seed=seed * 101 + a, max_tokens=max_tokens, temperature=temperature)
        stats.calls += 1
        try:
            obj, repaired = parse_json(raw)
            stats.repaired += int(repaired)
            return obj
        except ValueError as e:
            stats.parse_fail += 1
            last = e
    raise last


def mode_violation(text, mode, name):
    """Rule-based compliance check used by the Director. Returns a hint string or ''."""
    first = re.escape(name.split()[0])
    if mode == "first_person":
        if re.search(rf"\b{first}\w*\b[^.?!]{{0,40}}\b{REPORT_VERB}\b", text, re.I) and not SPEECH_PRON.search(text):
            return "it was narration about the character, not the character's own direct speech"
        if re.match(rf"\s*(the\s+)?{first}\w*\s+\w+(ed|s)\b", text, re.I):
            return "it described the character in the third person instead of giving the character's own words"
        return ""
    if mode.startswith("third_person"):
        if QUOTE.search(text):
            return "it contained quotation marks / direct speech"
        if SPEECH_PRON.search(text):
            return "it used first- or second-person pronouns (I, me, my, we, you, your); a narrator must report the turn in the third person"
        if not (re.search(rf"\b{first}", text, re.I) or THIRD_REF.search(text)):
            return "it did not refer to the character by name or he/she"
    return ""


class DirectorAgent:
    def __init__(self, llm, seed=0):
        self.llm, self.seed, self.stats = llm, seed, CallStats()

    def initialise(self, excerpt, cast=None):
        """cast: optional {canonical name: [aliases]} used to verify/merge the extracted speakers."""
        d = call_json(self.llm, prompts.EXTRACT_DIALOGUE.format(text=excerpt), self.seed, self.stats,
                      max_tokens=3000, temperature=0.0)
        raw = [x for x in d.get("dialogues", []) if isinstance(x, dict) and isinstance(x.get("text"), str)
               and x.get("text") and isinstance(x.get("speaker"), str) and x.get("speaker")]
        if cast:
            return self._initialise_with_cast(excerpt, raw, cast)
        dialogue = raw
        speakers = sorted({x["speaker"] for x in dialogue})
        # participants = speakers plus characters who are addressed (e.g. a silent listener)
        for x in dialogue:
            to = x.get("speak_to") or []
            for a in ([to] if isinstance(to, str) else to):
                if isinstance(a, str) and a.strip() and a.lower() not in GENERIC and not self._canon(a, speakers):
                    speakers.append(a.strip())
        speakers = sorted(set(speakers))
        c = call_json(self.llm, prompts.EXTRACT_CHARACTERS.format(text=excerpt, speakers=json.dumps(speakers),
                      dialogue=_fmt_history(dialogue)), self.seed, self.stats, max_tokens=3000, temperature=0.0)
        chars = {}
        for ch in c.get("characters", []):
            if not isinstance(ch, dict) or not ch.get("name"):
                continue
            name = self._canon(ch["name"], speakers) or ch["name"]
            chars[name] = dict(name=name, gender=ch.get("gender", "unknown"), personality=ch.get("personality", ""),
                               emotional_baseline=_num(ch.get("emotional_baseline", 0.0)),
                               speech_style=ch.get("speech_style", ""), description=ch.get("description", ""),
                               relationships=ch.get("relationships", {}))
        for s in speakers:  # every speaker gets a card, even if the LLM forgot one
            chars.setdefault(s, dict(name=s, gender="unknown", personality="", emotional_baseline=0.0,
                                     speech_style="", description="", relationships={}))
        names = list(chars)
        for x in dialogue:
            x["speaker"] = self._canon(x["speaker"], names) or x["speaker"]
            to = x.get("speak_to") or []
            to = [to] if isinstance(to, str) else to
            x["speak_to"] = [self._canon(t, names) or t for t in to if isinstance(t, str)]
        return dict(dialogue=dialogue, characters=chars, context=c.get("context", ""))

    def _initialise_with_cast(self, excerpt, raw, cast):
        names = list(cast)
        lookup = {}
        for n, aliases in cast.items():
            for a in [n] + aliases:
                lookup[_norm(a)] = n

        def to_cast(label):
            if not isinstance(label, str):
                return None
            k = _norm(label)
            if k in lookup:
                return lookup[k]
            m = difflib.get_close_matches(k, list(lookup), n=1, cutoff=0.8)
            if m:
                return lookup[m[0]]
            hits = [n for n in names if _norm(n) in k or k in _norm(n)]
            return hits[0] if len(hits) == 1 else None

        dialogue, dropped = [], 0
        for x in raw:
            sp = to_cast(x["speaker"])
            if sp is None:
                dropped += 1
                continue
            to = x.get("speak_to") or []
            to = [to] if isinstance(to, str) else to
            to = [m for m in dict.fromkeys(to_cast(a) for a in to) if m and m != sp]
            if not to:
                prev = [y["speaker"] for y in dialogue if y["speaker"] != sp]
                to = [prev[-1]] if prev else [n for n in names if n != sp][:1]
            dialogue.append(dict(speaker=sp, speak_to=to, text=x["text"]))
        c = call_json(self.llm, prompts.EXTRACT_CHARACTERS.format(text=excerpt, speakers=json.dumps(names),
                      dialogue=_fmt_history(dialogue)), self.seed, self.stats, max_tokens=3000, temperature=0.0)
        cards = {}
        for ch in c.get("characters", []):
            if isinstance(ch, dict) and to_cast(ch.get("name")) in names and to_cast(ch.get("name")) not in cards:
                n = to_cast(ch["name"])
                cards[n] = dict(name=n, gender=ch.get("gender", "unknown"), personality=ch.get("personality", ""),
                                emotional_baseline=_num(ch.get("emotional_baseline", 0.0)), speech_style=ch.get("speech_style", ""),
                                description=ch.get("description", ""), relationships=ch.get("relationships", {}))
        for n in names:
            cards.setdefault(n, dict(name=n, gender="unknown", personality="", emotional_baseline=0.0,
                                     speech_style="", description="", relationships={}))
        stats = dict(raw_utterances=len(raw), kept=len(dialogue), dropped_unmapped=dropped,
                     raw_labels=sorted({x["speaker"] for x in raw}))
        return dict(dialogue=dialogue, dialogue_raw=raw, characters=cards, context=c.get("context", ""), cast_mapping=stats)

    def rewrite_third(self, dialogue):
        out = call_json(self.llm, prompts.REWRITE_THIRD.format(n=len(dialogue), dialogue=json.dumps(dialogue, ensure_ascii=False, indent=1)),
                        self.seed, self.stats, max_tokens=3000, temperature=0.0)
        rew = out.get("dialogues", [])
        if len(rew) != len(dialogue):
            rew = (rew + dialogue)[:len(dialogue)]
        return [dict(speaker=o["speaker"], speak_to=o.get("speak_to", []), text=(r.get("text") or o["text"]))
                for o, r in zip(dialogue, rew)]

    def render(self, text, mode, name, to, seed):
        """Enforcement fallback: rewrite a non-compliant turn into the required narrative form."""
        p = prompts.DIRECTOR_RENDER.format(name=name, to=", ".join(to) or "the others", text=text,
                                           mode_block=prompts.MODE_BLOCKS[mode].format(name=name))
        out = call_json(self.llm, p, seed, self.stats, temperature=0.0)
        return out.get("content") if isinstance(out.get("content"), str) else None

    def narrate(self, speech, name, to, seed, hint=""):
        """Speak-then-narrate: turn an Actor's direct speech into third-person free indirect narration,
        changing only pronouns, tense and at most a short reporting phrase."""
        p = prompts.DIRECTOR_NARRATE.format(name=name, to=", ".join(to) or "the others", text=speech)
        if hint:
            p += prompts.CORRECTION.format(hint=hint)
        out = call_json(self.llm, p, seed, self.stats, temperature=0.0)
        return out.get("content") if isinstance(out.get("content"), str) else None

    @staticmethod
    def _canon(name, names):
        if not isinstance(name, str):
            return None
        low = {n.lower(): n for n in names}
        if name.lower() in low:
            return low[name.lower()]
        m = difflib.get_close_matches(name.lower(), list(low), n=1, cutoff=0.75)
        return low[m[0]] if m else None


class ActorAgent:
    def __init__(self, llm, card, all_names, exemplars=None):
        self.llm, self.card = llm, card
        self.others = [n for n in all_names if n != card["name"]]
        self.stats = CallStats()
        self.exemplars = ""
        if exemplars:  # optional voice exemplars: verbatim source lines of this character
            lines = "\n".join(f'- "{x}"' for x in exemplars)
            self.exemplars = prompts.EXEMPLARS.format(name=card["name"], lines=lines)

    def act(self, history, context, mode, seed, hint=""):
        c = self.card
        mode_block = prompts.MODE_BLOCKS[mode].format(name=c["name"])
        if hint:
            mode_block += prompts.CORRECTION.format(hint=hint)
        prompt = prompts.ACTOR.format(
            name=c["name"], personality=c["personality"], speech_style=c["speech_style"],
            description=c["description"], relationships=json.dumps(c["relationships"], ensure_ascii=False),
            emotion=f"{c['emotional_baseline']:.2f}", context=context, others=", ".join(self.others),
            exemplars=self.exemplars,
            history=_fmt_history(history), mode_block=mode_block,
            content_desc=prompts.CONTENT_DESC[mode].format(name=c["name"]))
        return call_json(self.llm, prompt, seed, self.stats)


def single_prompt_session(llm, scene, opening, mode, n_turns, seed):
    """Baseline: one LLM call writes the whole continuation (no agents, no memory control)."""
    stats = CallStats()
    cards = "\n".join(f"- {c['name']}: {c['personality']} Speech style: {c['speech_style']}" for c in scene["characters"].values())
    p = prompts.SINGLE_PROMPT.format(n=n_turns, names=", ".join(scene["characters"]), cards=cards,
                                     context=scene["context"], history=_fmt_history(opening),
                                     mode_text=prompts.SINGLE_MODE_TEXT[mode])
    obj = call_json(llm, p, seed, stats, max_tokens=4000)
    turns = []
    for t in obj.get("turns", [])[:n_turns]:
        if isinstance(t, dict) and t.get("text"):
            sp = DirectorAgent._canon(t.get("speaker", ""), list(scene["characters"])) or t.get("speaker", "")
            to = t.get("speak_to") or []
            turns.append(dict(speaker=sp, speak_to=[to] if isinstance(to, str) else to, text=t["text"]))
    return turns, stats.as_dict()


def _norm(s):
    return re.sub(r"[^a-z0-9\- ]", "", s.lower().replace("’", "'")).strip()


def _num(x):
    try:
        return max(-1.0, min(1.0, float(x)))
    except Exception:
        return 0.0
