"""Prompt templates. All templates are fixed strings; only the role card, the scene
context, the (truncated) dialogue history and the narrative-mode block are filled in."""

EXTRACT_DIALOGUE = """Extract the direct-speech dialogue from the following literary excerpt, in order of appearance.
Rules:
1. Include only spoken utterances (no narration, inner thoughts or action descriptions).
2. If an utterance is interrupted by narration, merge the parts into one utterance.
3. "speaker": use the character's name if it is known from the excerpt; otherwise the first descriptive reference (e.g. "the old lady"). Use exactly the same label for the same person throughout. Never invent labels such as "Speaker A".
4. "speak_to": always a non-empty array of addressee labels (use the same labels as for speakers); infer the addressee from context if it is not explicit.
Return only JSON of the form {{"dialogues": [{{"speaker": "...", "speak_to": ["..."], "text": "..."}}]}}.

Excerpt:
{text}
"""

EXTRACT_CHARACTERS = """You are the Director of a dialogue simulation. Based on the literary excerpt and its extracted dialogue, build a role card for every character who speaks.
Return only JSON of the form:
{{"characters": [{{"name": "...", "gender": "female|male|unknown", "personality": "...", "emotional_baseline": 0.0, "speech_style": "...", "description": "...", "relationships": {{"Other name": "relation"}}}}],
  "context": "two or three sentences summarising the situation of the scene"}}
"emotional_baseline" is a number in [-1, 1]. Use exactly these speaker labels as names: {speakers}.

Excerpt:
{text}

Extracted dialogue:
{dialogue}
"""

THIRD_EXAMPLE = ('Example. Direct speech: John said to Mary, "I\'m tired and don\'t want to go to the party." '
                 "Third-person narration: John told Mary that he was tired and had no intention of going to the party.")

REWRITE_THIRD = """Rewrite each utterance below as third-person narration by an external narrator, in the style of a novel (reported speech).
Each item must begin with or contain the speaker's name and a reporting verb (e.g. told, asked, insisted, replied), must not contain quotation marks, and must not contain the pronouns I, me, my, we, us, our, you or your. Keep the meaning, the speaker and the addressees unchanged.
""" + THIRD_EXAMPLE + """
Return only JSON of the form {{"dialogues": [{{"speaker": "...", "speak_to": ["..."], "text": "..."}}]}} with exactly {n} items in the same order.

Utterances:
{dialogue}
"""

MODE_BLOCKS = {
    "first_person": (
        "NARRATIVE MODE: FIRST PERSON. Write {name}'s turn as direct speech in {name}'s own voice. "
        "{name} refers to himself or herself with first-person pronouns (I, me, my). "
        "Do not add a narrator, quotation marks, or descriptions of {name} in the third person."
    ),
    "third_person": (
        "NARRATIVE MODE: THIRD PERSON. Do not write direct speech. Write {name}'s turn as narration by an external narrator, "
        "as in a novel: report what {name} says (and, if relevant, does or feels), naming {name} and using a reporting verb "
        "(told, asked, insisted, ...). Never use I, me, my, we, our, you or your, and no quotation marks. " + THIRD_EXAMPLE
    ),
    "neutral": "",  # control condition: no perspective instruction at all
    # improvement experiment: third-person narration that keeps the character's own idiom (free indirect style)
    "third_person_voice": (
        "NARRATIVE MODE: THIRD PERSON, VOICE-PRESERVING. Do not write direct speech. Write {name}'s turn as narration by an "
        "external narrator, referring to {name} by name or he/she, never as \"I\", and use no quotation marks or the pronouns "
        "I, me, my, we, our, you or your. Keep {name}'s own words, idioms, tone and characteristic expressions as far as possible "
        "(free indirect style): report the turn in {name}'s voice instead of paraphrasing it in the narrator's formal language. "
        "Example. Direct speech: John said to Mary, \"I'm dead tired and I won't go to that silly party.\" Voice-preserving narration: "
        "John was dead tired, he told Mary, and he would not go to that silly party."
    ),
}

# improvement experiment 2 (speak-then-narrate): the Actor speaks in its own voice (first-person block),
# the Director then narrates the speech, changing only what the narrative form requires
MODE_BLOCKS["third_person_narrated"] = MODE_BLOCKS["first_person"]

DIRECTOR_NARRATE = """You are the narrator of a novel. Turn the following direct speech of {name} (addressed to {to}) into third-person narration in free indirect style.
Rules: keep {name}'s own words, idioms, imagery and tone; change only what the narrative form requires: first-person pronouns (I, me, my, we, us, our) become {name}, he, she or they; second-person pronouns (you, your) become the addressee's name, he, she or they; verbs shift to the past tense. Do not use quotation marks or the pronouns I, me, my, we, us, our, you or your. Do not add commentary, interpretations or feelings that are not in the speech; at most add a short reporting phrase such as "he told Mary".
Example. Direct speech of John to Mary: I'm dead tired and I won't go to that silly party. Narration: John was dead tired, he told Mary, and he would not go to that silly party.
Speech: {text}
Return only JSON: {{"content": "..."}}
"""

CONTENT_DESC = {
    "third_person_narrated": "{name}'s words as direct speech (empty if silent)",
    "third_person_voice": "the narrator's third-person report of {name}'s turn in {name}'s own idiom (empty if silent)",
    "first_person": "{name}'s words as direct speech (empty if silent)",
    "third_person": "the narrator's third-person report of {name}'s turn (empty if silent)",
    "neutral": "{name}'s turn (empty if silent)",
}

DIRECTOR_RENDER = """You are the Director of a dialogue simulation and you enforce the narrative mode.
Rewrite the following turn of {name} (addressed to {to}) so that it follows the mode below. Keep its meaning, tone and length; change only the narrative form.
{mode_block}
Turn: {text}
Return only JSON: {{"content": "..."}}
"""

ACTOR = """You are role-playing the literary character {name} in a multi-character scene.

Role card:
- Name: {name}
- Personality: {personality}
- Speech style: {speech_style}
- Background: {description}
- Relationships: {relationships}
- Current emotion (-1 negative .. 1 positive): {emotion}

{exemplars}Scene: {context}
Other characters present: {others}

Recent dialogue (oldest first):
{history}

Instructions: Stay in character. Decide what {name} says next, consistent with the role card and the recent dialogue. Advance the scene; do not repeat earlier turns. Keep the turn to one to three sentences.
{mode_block}
Return only JSON: {{"intend": "speak" or "silent", "content": "{content_desc}", "speak_to": ["addressee name(s)"]}}
"""

EXEMPLARS = ("Voice examples (lines that {name} speaks in the source text; imitate the voice and diction, "
             "not the content):\n{lines}\n\n")

CORRECTION ="\nIMPORTANT: a previous answer did not follow the narrative mode ({hint}). Follow the NARRATIVE MODE exactly.\n"

SINGLE_PROMPT = """Write the continuation of the following literary scene as a dialogue of exactly {n} turns among these characters: {names}.

Role cards:
{cards}

Scene: {context}

Opening of the scene:
{history}
{mode_text}
Each turn is one to three sentences. Return only JSON of the form {{"turns": [{{"speaker": "...", "speak_to": ["..."], "text": "..."}}]}} with {n} items.
"""

SINGLE_MODE_TEXT = {
    "first_person": "Every turn must be direct speech in the speaker's own voice using first-person pronouns (I, me, my); no narrator and no quotation marks.",
    "third_person": "Every turn must be third-person narration by an external narrator reporting what the speaker says (by name or he/she, never 'I'), with no direct speech or quotation marks.",
    "neutral": "",
}
