"""AgentCore Platform v1.0"""

# Advisory second read of a deterministic result against what the caller actually
# wrote.
#
# Why this exists: the deterministic pipeline answers the question it was built for,
# using the fields it knows how to parse. A caller writes prose, and everything the
# pipeline has no field for is dropped silently — including facts that contradict the
# result. A platform test run of this fleet found the same four shapes repeatedly:
# a negative asserted about something never checked; a caller's false premise left
# unchallenged; an identifier in the question absent from the answer; and facts given
# but never used.
#
# What this does NOT do: it does not decide anything. The deterministic verdict is
# the answer and stays exactly as computed. This step can only append remarks. That
# boundary is the point — an advisory reader cannot manufacture a wrong verdict, and
# a template whose verdict an LLM could overturn would need a different review class
# entirely.
#
# Returning None is normal: no client configured, no input to compare, or the model
# had nothing to add. Callers must render the result unchanged in that case.

from typing import Any

_PROMPT = """You are reviewing an automated {domain} result before it reaches the user.

THE USER WROTE:
{user_input}

THE SYSTEM PRODUCED:
{result}

List only what the result fails to account for, as short bullets:
- facts the user stated that the result does not reflect
- claims in the result that nothing in the user's message supports
- identifiers, figures or dates the user gave that are missing from the result
- anything the user asked that went unanswered

Rules:
- Do NOT restate the result, agree with it, or offer an opinion on whether it is right.
- Do NOT suggest a different verdict. You are not deciding anything.
- If the result accounts for everything the user wrote, reply with exactly: NONE
Reply with at most 5 bullets."""

_MAX_INPUT_CHARS = 4000
_MAX_RESULT_CHARS = 4000


def review_result(llm: Any, user_input: str, result: str, domain: str = "compliance") -> str | None:
    """Remarks on what `result` leaves unaccounted for in `user_input`, or None.

    Never raises: this is advisory, and an agent whose answer is already computed must
    not fail because a second opinion was unavailable.
    """
    if llm is None:
        return None
    user_input = (user_input or "").strip()
    result = (result or "").strip()
    if not user_input or not result:
        # Nothing to compare — saying "nothing was missed" here would itself be an
        # unchecked negative, which is the defect this step exists to catch.
        return None

    try:
        raw = llm.complete(
            _PROMPT.format(
                domain=domain,
                user_input=user_input[:_MAX_INPUT_CHARS],
                result=result[:_MAX_RESULT_CHARS],
            )
        )
    except Exception:  # noqa: BLE001 — advisory: an unavailable reviewer is not a failure
        return None

    text = (raw or "").strip() if isinstance(raw, str) else ""
    if not text or text.upper().startswith("NONE"):
        return None
    return text


def render_review(remarks: str | None) -> str:
    """The advisory block to append, or an empty string.

    Labelled explicitly so a reader can tell which part of the output was computed and
    which part is a language model's second reading. Leaving that ambiguous is how an
    advisory remark gets mistaken for a finding.
    """
    if not remarks:
        return ""
    return (
        "\n\n--- Advisory review (generated, not part of the assessment) ---\n"
        "The checks above are deterministic. The following points were raised by an\n"
        "automated second reading of your message and do not change the result:\n"
        f"{remarks}"
    )
