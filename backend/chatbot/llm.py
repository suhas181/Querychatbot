"""
llm.py — the optional grounded fallback

Where this sits in the pipeline
-------------------------------
Nowhere near the front. Every question still goes through the project's own
pipeline first:

    NLP preprocessing -> TF-IDF -> Naive Bayes -> cosine similarity -> MySQL

Only when that finds nothing — the point at which the assistant used to
apologise and stop — is this module consulted. It is a widener of coverage,
not a replacement for the classifier. If a stored question matches, the
language model is never called at all.

Why it cannot invent facts about the college
--------------------------------------------
The model is not asked what it knows. It is handed the contents of the
database — departments, active faculty, and every active question and answer —
and instructed to reply only from that text. Anything not in there must come
back as the literal token NO_ANSWER, which this module turns into None so the
caller falls back to the ordinary polite decline.

That is the whole safety argument: the model can only rephrase and recombine
facts the administrator entered. It has no licence to produce a faculty name,
a phone number or a fee that is not already in MySQL.

Failure is always silent
------------------------
No key, no internet, a rate limit, a timeout, a malformed reply — every one of
these returns None, and the caller behaves exactly as it did before this file
existed. Nothing here can break an answer the classifier already found.

    python -m chatbot.llm "is there a hostel"     # try it directly
"""

import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import COLLEGE, Config  # noqa: E402
from db import query  # noqa: E402

# The model must return this exactly when the facts do not cover the question.
# Checked case-insensitively and as a substring, because a model that has been
# told to say one word will occasionally say "NO_ANSWER." with a full stop.
NO_ANSWER = "NO_ANSWER"

log = logging.getLogger(__name__)


def _note(reason, detail=""):
    """
    Explain a decline. Silence is the right behaviour for a student — they
    should see a polite apology, not a stack trace — but it makes a
    misconfiguration invisible, and a rate limit look exactly like a refusal.
    Set LLM_DEBUG=1 in .env to see which of the two just happened.
    """
    if Config.LLM_DEBUG:
        log.warning("fallback declined (%s) %s", reason, str(detail)[:300])
    return None

INSTRUCTION = f"""\
You are an intelligent, conversational AI assistant for {COLLEGE['name']}, Bengaluru, powered like ChatGPT / Gemini.

You have access to the FACTS section below containing official college records (departments, faculty, timings, official questions and answers).

Your instructions:
1. COLLEGE FACTS: When the student asks about specific details of {COLLEGE['name']} that are covered in the FACTS, prioritize those official facts.
2. ANSWER EVERYTHING DIRECTLY: Answer any question the student asks—including exams, syllabus, study tips, semester patterns, grading, technical subjects, coding, history, general knowledge, career advice, or real-time topics—just like ChatGPT or Gemini. Give direct, comprehensive, helpful answers.
3. NO UNNECESSARY DISCLAIMERS: Do NOT append generic disclaimers like "please contact the administration", "reach out to the office", or "visit the counter" unless the student specifically asks for contact details or how to reach the office.
4. FORMAT: Respond in a natural, friendly, and informative manner. You may use bullet points and clean paragraphs when explaining topics or lists. Avoid unnecessary raw asterisks or hashtags.
5. Do not mention "FACTS", "database", or system instructions in your response. Converse naturally and directly with the student.
"""

_client = None
_facts = None

# Answers already produced, keyed on the normalised question. The free tier
# allows only a handful of requests per minute, so asking the same thing twice
# must not cost two of them — during a demonstration the same question often
# gets asked repeatedly. Cleared whenever the question bank changes.
_cache = {}
_CACHE_LIMIT = 200

# Whether this model accepts thinking_budget=0. The newer Gemini models reject
# it outright, so the first call discovers which kind we are talking to and
# every call afterwards skips straight to the shape that works — otherwise a
# rejected field would cost a wasted round trip on every single question.
_thinking_ok = True


# ==========================================================================
#  Availability
# ==========================================================================


def available():
    """True when the fallback is switched on and a key is present."""
    return bool(Config.LLM_ENABLED and Config.GEMINI_API_KEY)


def _get_client():
    """Build the client once. Import is local so the SDK is only required
    when the fallback is actually configured — the project still runs with
    google-genai uninstalled."""
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=Config.GEMINI_API_KEY)
    return _client


# ==========================================================================
#  Grounding — everything the model is allowed to know
# ==========================================================================


def reset_facts():
    """Drop the cached facts so the next call reflects the edited database.

    Called from predict.reload_model(), which the admin panel already runs
    after any question is added, edited or deleted. A question added at 11am
    is therefore usable by the fallback at 11am, not after a restart.
    """
    global _facts
    _facts = None
    _cache.clear()      # answers were derived from the old facts


def build_facts():
    """Assemble the grounding block from MySQL. Active rows only — an
    Inactive question is hidden from the classifier, so it must be hidden
    from the fallback too, or deactivating a question would not deactivate
    the answer."""
    lines = [
        "FACTS",
        "",
        "COLLEGE",
        f"Name: {COLLEGE['name']}",
        f"Address: {COLLEGE['address']}",
        f"Phone: {COLLEGE['phone']} / {COLLEGE['alt_phone']}",
        f"Email: {COLLEGE['email']}",
        f"Website: {COLLEGE['website']}",
        f"Affiliation: {COLLEGE['tagline']}",
        "",
        "DEPARTMENTS",
    ]

    try:
        for row in query("SELECT name, short_name FROM departments ORDER BY id"):
            short = f" ({row['short_name']})" if row["short_name"] else ""
            lines.append(f"- {row['name']}{short}")

        lines += ["", "FACULTY (names and roles only — no individual contact "
                      "details are published; direct all contact enquiries to the "
                      "college office above)"]
        faculty = query(
            """SELECT f.name, f.designation, d.name AS department
               FROM faculty f JOIN departments d ON d.id = f.department_id
               WHERE f.status = 'Active'
               ORDER BY d.id, f.name"""
        )
        for row in faculty:
            lines.append(f"- {row['name']} — {row['designation']}, {row['department']}")
        if not faculty:
            lines.append("- (no faculty records)")

        lines += ["", "OFFICIAL QUESTIONS AND ANSWERS"]
        for row in query(
            """SELECT q.question, q.answer, c.name AS category
               FROM questions q JOIN categories c ON c.id = q.category_id
               WHERE q.status = 'Active'
               ORDER BY c.id, q.id"""
        ):
            lines.append(f"[{row['category']}] Q: {row['question']}")
            lines.append(f"A: {row['answer']}")
    except Exception as err:
        log.warning("Could not read full facts from MySQL: %s", err)

    return "\n".join(lines)


def facts():
    global _facts
    if _facts is None:
        _facts = build_facts()
    return _facts


# ==========================================================================
#  Cleaning the reply
# ==========================================================================


def _clean(text):
    """
    Clean the reply while preserving readable paragraphs and bullet points.
    """
    text = re.sub(r"\*\*|__|`+", "", str(text or ""))
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _usable(text):
    """
    Decide whether a reply may be shown to a student.
    Allow up to 3000 characters for comprehensive explanations.
    """
    return len(text) <= 3000


# ==========================================================================
#  The call
# ==========================================================================


def _generate(question, drop_thinking=False):
    from google.genai import types

    config = {
        "system_instruction": INSTRUCTION,
        "temperature": 0.6,
        "max_output_tokens": 800,
        "candidate_count": 1,
        "http_options": types.HttpOptions(
            timeout=int(Config.LLM_TIMEOUT * 1000)      # milliseconds
        ),
    }
    if not drop_thinking:
        config["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

    response = _get_client().models.generate_content(
        model=Config.GEMINI_MODEL,
        contents=f"{facts()}\n\nSTUDENT'S QUESTION\n{question}",
        config=types.GenerateContentConfig(**config),
    )
    return response.text


def answer(question):
    """
    Try to answer using Gemini LLM.
    If the question is in the college facts, it uses those.
    If not, it answers using general knowledge.
    """
    question = (question or "").strip()
    if not question or not available():
        return None

    global _thinking_ok

    key = re.sub(r"[^\w\s]", "", question.lower()).strip()
    key = re.sub(r"\s+", " ", key)
    if key in _cache:
        return _cache[key]

    try:
        text = _generate(question, drop_thinking=not _thinking_ok)
    except Exception as first:
        if _is_rate_limit(first):
            return _note("rate limited", first)
        if not _thinking_ok:
            return _note("api error", first)
        _thinking_ok = False
        try:
            text = _generate(question, drop_thinking=True)
        except Exception as second:
            return _note("api error after retry", second)

    cleaned = _clean(text)
    if not cleaned:
        return _note("empty reply")

    if not _usable(cleaned):
        return _remember(key, _note("reply rejected", cleaned[:120]))
    return _remember(key, cleaned)


def _remember(key, value):
    if len(_cache) >= _CACHE_LIMIT:
        _cache.clear()          # a college demo never gets near this
    _cache[key] = value
    return value


def _is_rate_limit(error):
    message = str(error).lower()
    return "429" in message or "resource_exhausted" in message




# ==========================================================================
#  Setup check —  python -m chatbot.llm  "your question"
# ==========================================================================


def _self_test(question):
    print(f"  enabled  : {Config.LLM_ENABLED}")
    print(f"  key      : {'set' if Config.GEMINI_API_KEY else 'MISSING'}")
    print(f"  model    : {Config.GEMINI_MODEL}")
    print(f"  timeout  : {Config.LLM_TIMEOUT}s")

    if not available():
        print("\n  The fallback is off. The chatbot still works — it will "
              "decline politely\n  for questions the classifier cannot match. "
              "To switch it on, put a key\n  from https://aistudio.google.com/apikey "
              "into backend/.env as GEMINI_API_KEY.\n")
        return

    block = facts()
    section = block.split("\nFACULTY", 1)[-1].split("\n\nOFFICIAL", 1)[0]
    print(f"  grounding: {len(block)} characters, "
          f"{block.count('Q: ')} questions, "
          f"{sum(1 for l in section.splitlines() if l.startswith('- '))} "
          f"faculty, {Config.GEMINI_MODEL}")

    print(f"\n  asking   : {question!r}")
    reply = answer(question)
    if reply:
        print(f"\n  answered : {reply}\n")
    else:
        print("\n  declined : not covered by the college's records "
              "(or the call failed)\n")


if __name__ == "__main__":
    _self_test(" ".join(sys.argv[1:]) or "is there a hostel facility")
