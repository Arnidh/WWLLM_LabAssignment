"""
Prompts for grounded lecture notes, chunk reduction, quizzes, and translation.
"""
from __future__ import annotations


SYSTEM_GROUNDING = (
    "You are a precise academic assistant. Use ONLY information explicitly "
    "present in the supplied lecture transcript or notes. Do not add outside "
    "facts or make unsupported inferences. When information is missing or "
    "unclear, say so explicitly instead of guessing."
)

NOTES_SCHEMA_DESCRIPTION = """\
Return one JSON object with exactly these fields:
{
  "title": "short title",
  "key_points": ["5 to 10 concise points"],
  "key_terms": [{"term": "term", "definition": "one-line transcript-grounded definition"}],
  "summary": "short summary in 2-4 sentences",
  "missing_information": ["important facts that are missing or unclear; [] if none"]
}
All values must be written in the requested lecture language. Do not add Markdown fences.
"""

NOTES_FEW_SHOT = """\
Example:
Transcript: "Today we cover Newton's second law: force equals mass times acceleration."
JSON:
{"title":"Newton's Second Law","key_points":["Force equals mass times acceleration."],"key_terms":[{"term":"Force","definition":"The lecture states that force equals mass times acceleration."}],"summary":"The lecture introduces Newton's second law as force equals mass times acceleration.","missing_information":[]}
"""


def notes_prompt(transcript: str, lang_name: str) -> str:
    return f"""\
{NOTES_FEW_SHOT}

Create structured lecture notes in {lang_name} from the transcript below.
Use ONLY the transcript. Return 5-10 key points, one-line definitions for terms
that the transcript explains, and a short 2-4 sentence summary. List missing or
unclear information in "missing_information"; do not fill gaps with guesses.

{NOTES_SCHEMA_DESCRIPTION}
Transcript:
\"\"\"
{transcript}
\"\"\"
JSON:
"""


def notes_chunk_summary_prompt(chunk: str, lang_name: str) -> str:
    return f"""\
Summarize this transcript group as concise factual bullet points in {lang_name}.
Use ONLY facts stated in the group. Preserve important names, numbers, and
uncertainties. Do not add outside knowledge; note when information is missing.

Transcript group:
\"\"\"
{chunk}
\"\"\"

Factual summary:
"""


def notes_combine_prompt(summaries: str, lang_name: str) -> str:
    return f"""\
Combine these consecutive transcript-group summaries into complete structured
lecture notes in {lang_name}. Deduplicate repeated points and preserve distinct
facts. Do not infer missing information; explicitly identify uncertainty.

{NOTES_SCHEMA_DESCRIPTION}
Group summaries:
\"\"\"
{summaries}
\"\"\"
JSON:
"""


MCQ_SCHEMA_DESCRIPTION = """\
Return a JSON array with exactly 5 objects. Each object has exactly:
- "question": non-empty string
- "options": object with exactly "A", "B", "C", "D", each a distinct non-empty string
- "answer": exactly one of "A", "B", "C", "D"
- "explanation": one concise sentence grounded in the source
- "evidence": one complete sentence copied verbatim from the supplied transcript evidence
"""

MCQ_FEW_SHOT = """\
Example:
Transcript evidence: "Force equals mass times acceleration."
Question source: "The lecture states that force equals mass times acceleration."
JSON:
[{"question":"According to the lecture, what does force equal?","options":{"A":"Mass times acceleration","B":"Mass plus velocity","C":"Acceleration divided by mass","D":"Velocity times time"},"answer":"A","explanation":"The lecture directly states that force equals mass times acceleration.","evidence":"Force equals mass times acceleration."}]
"""


def mcq_prompt(
    source_text: str,
    lang_name: str,
    evidence_text: str | None = None,
) -> str:
    evidence = evidence_text if evidence_text is not None else source_text
    return f"""\
{MCQ_FEW_SHOT}

Generate exactly 5 different multiple-choice questions in {lang_name}, based
ONLY on the question source. Do not ask for facts absent from the source. There
must be four distinct options (A-D) and exactly one correct answer. The
explanation must be one sentence. For "evidence", copy a complete supporting
sentence exactly from the transcript evidence below; do not paraphrase it.
Return ONLY the JSON array, with no prose or Markdown fences.

{MCQ_SCHEMA_DESCRIPTION}
Question source:
\"\"\"
{source_text}
\"\"\"

Transcript evidence:
\"\"\"
{evidence}
\"\"\"

JSON:
"""


def mcq_repair_prompt(
    bad_json: str,
    error_msg: str,
    source_text: str,
    evidence_text: str,
    lang_name: str,
) -> str:
    return f"""\
The previous quiz response failed validation:
{error_msg}

Correct the response using ONLY the question source below. Keep all evidence
verbatim from the transcript evidence. Do not invent, paraphrase, or shorten
evidence. If a question has no supported evidence, replace that question with
one that does. Return only a JSON array of exactly 5 valid questions in
{lang_name}; no prose or Markdown fences.

{MCQ_SCHEMA_DESCRIPTION}
Question source:
\"\"\"
{source_text}
\"\"\"

Transcript evidence:
\"\"\"
{evidence_text}
\"\"\"

Previous response:
\"\"\"
{bad_json}
\"\"\"

Corrected JSON:
"""


def translation_prompt(text: str, src_lang: str, tgt_lang: str) -> str:
    """Fallback prompt for a future text-based translation path."""
    return f"""\
Translate the following text from {src_lang} to {tgt_lang}.
Output ONLY the translation; do not add explanations or facts.

Text:
\"\"\"
{text}
\"\"\"

Translation:
"""
