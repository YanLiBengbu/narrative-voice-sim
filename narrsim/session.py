"""Multi-agent dialogue session (Algorithm 1 of the paper): seeded speaker selection,
sliding-window memory of the last k turns, Director-enforced narrative mode."""

import numpy as np

from .agents import ActorAgent, CallStats, DirectorAgent, mode_violation


def speaker_scores(names, history):
    """Addressee-aware attention scores f(e_i, H'): prefer characters addressed in the last
    turn, penalise the previous speaker, and favour characters who have not spoken recently."""
    last = history[-1] if history else None
    addressed = set(last.get("speak_to") or []) if last else set()
    k = max(len(names), 1)
    scores = []
    for n in names:
        gap = k
        for j, t in enumerate(reversed(history)):
            if t["speaker"] == n:
                gap = min(j, k)
                break
        s = 2.0 * (n in addressed) - 1.5 * (last is not None and last["speaker"] == n) + 1.0 * gap / k
        scores.append(s)
    return np.array(scores, dtype=float)


def select_speaker(policy, names, history, rng, tau=0.7):
    """Seeded speaker sampling; 'round_robin' is handled directly in run_session."""
    if policy == "uniform":
        return names[int(rng.integers(len(names)))]
    s = speaker_scores(names, history) / tau
    p = np.exp(s - s.max())
    p /= p.sum()
    return names[int(rng.choice(len(names), p=p))]


def run_session(llm, scene, opening, mode, seed, max_turns=20, memory_k=5, policy="addressee",
                enforce=True, max_consecutive_silent=4, exemplars=False, n_exemplars=3):
    names = list(scene["characters"])
    ex = {n: [] for n in names}
    if exemplars:  # verbatim source lines of each character (from the Director's extraction)
        for t in scene["dialogue"]:
            if t["speaker"] in ex and len(ex[t["speaker"]]) < n_exemplars:
                ex[t["speaker"]].append(t["text"])
    actors = {n: ActorAgent(llm, scene["characters"][n], names, exemplars=ex[n]) for n in names}
    director = DirectorAgent(llm, seed=seed)
    rng = np.random.default_rng(seed)
    history = [dict(t) for t in opening]
    turns, silent, attempts = [], 0, 0
    rr = (names.index(history[-1]["speaker"]) + 1) if history and history[-1]["speaker"] in names else 0
    reason = "max_turns"
    while len(turns) < max_turns:
        if attempts >= 3 * max_turns:
            reason = "attempt_budget"
            break
        if policy == "round_robin":
            speaker = names[rr % len(names)]
            rr += 1
        else:
            speaker = select_speaker(policy, names, history, rng)
        window = history[-memory_k:] if memory_k > 0 else []
        turn_seed = seed * 1000 + attempts
        attempts += 1
        try:
            out = actors[speaker].act(window, scene["context"], mode, turn_seed)
        except ValueError:
            out = {"intend": "silent"}
        content = out.get("content").strip() if isinstance(out.get("content"), str) else ""
        if out.get("intend") != "speak" or not content:
            silent += 1
            if silent >= max_consecutive_silent:
                reason = "silence"
                break
            continue
        silent = 0
        to = out.get("speak_to") or []
        to = [to] if isinstance(to, str) else [x for x in to if isinstance(x, str)]
        flags = dict(compliant_first=True, regenerated=False, director_rewrite=False)
        speech = None
        if mode == "third_person_narrated":  # speak-then-narrate: the Director narrates the Actor's speech
            speech, narrated, ok = content, None, False
            for k, h in enumerate(("", "it still contained quotation marks or first/second-person pronouns")):
                try:
                    narrated = director.narrate(speech, speaker, to, turn_seed + k, hint=h)
                except ValueError:
                    narrated = None
                if narrated and not mode_violation(narrated, mode, speaker):
                    ok = True
                    break
                flags["compliant_first"] = False
            if ok:  # narrated at the first attempt, or after one re-narration with a hint
                content, flags["regenerated"] = narrated.strip(), not flags["compliant_first"]
            else:  # last resort: standard third-person rendering by the Director
                try:
                    c3 = director.render(speech, "third_person", speaker, to, turn_seed) if enforce else None
                except ValueError:
                    c3 = None
                content = (c3 or narrated or speech).strip()
                flags["director_rewrite"] = bool(c3)
            hint = ""
        else:
            hint = mode_violation(content, mode, speaker)
        if hint:
            flags["compliant_first"] = False
            if enforce:
                try:  # 1) ask the actor again with an explicit hint
                    out2 = actors[speaker].act(window, scene["context"], mode, turn_seed + 7919, hint=hint)
                    c2 = out2.get("content").strip() if isinstance(out2.get("content"), str) else ""
                except ValueError:
                    c2 = ""
                if c2 and not mode_violation(c2, mode, speaker):
                    content, flags["regenerated"] = c2, True
                else:  # 2) the Director rewrites the turn into the required form
                    try:
                        c3 = director.render(content, mode, speaker, to, turn_seed)
                    except ValueError:
                        c3 = None
                    if c3:
                        content, flags["director_rewrite"] = c3.strip(), True
        flags["residual_violation"] = bool(mode_violation(content, mode, speaker))
        rec = dict(turn=len(turns) + 1, speaker=speaker, speak_to=to, text=content, **flags)
        if speech is not None:
            rec["speech"] = speech  # the Actor's own words before narration
        turns.append(rec)
        history.append(dict(speaker=speaker, speak_to=to, text=content))
    agg = CallStats()
    for st in [a.stats for a in actors.values()] + [director.stats]:
        for k, v in st.as_dict().items():
            setattr(agg, k, getattr(agg, k) + v)
    return dict(turns=turns, termination=reason, attempts=attempts, call_stats=agg.as_dict())
