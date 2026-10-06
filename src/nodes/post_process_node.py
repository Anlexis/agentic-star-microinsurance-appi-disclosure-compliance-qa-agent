"""AgentCore Platform v1.0"""

# INS-C2-038 — PostProcessNode
#
# The outer backbone's post_process slot, and the template's output boundary. It
# is the last node that can withhold anything, so everything it does is written
# to fail closed.
#
# Three properties of the envelope shape this node, and each one is a way a
# careful-looking gate still discloses:
#
#   1. the base envelope resolves `formatted_output or result`, with no status
#      check — so returning a failed status while leaving `result` in state ships
#      the un-gated answer inside the failure envelope. On a violation this node
#      therefore CLEARS every field carrying answer text, not only the one it was
#      about to publish;
#   2. a FALSY replacement re-opens that same fallback. `""`, `{}` and an absent
#      key all fall through to `result`. The withheld notice is a non-empty
#      string, and the boundary tests assert it is present rather than asserting
#      the published field is empty;
#   3. the framework's own output gate raises when a credential shape appears in
#      ANY value this node returns, and the wrapper then discards the whole delta
#      — including the clearing. A local pattern set narrower than the framework's
#      is therefore a bypass, not a smaller net: the value passes here, the
#      framework raises, and the containment is thrown away with everything else.
#      The credential scan delegates to the framework's own detector for exactly
#      that reason.
#
# The node also refuses to fail with an exception. An unhandled failure here
# returns a delta with no formatted_output at all, which is case 1 again with no
# notice attached — so a malformed answer channel is handled as a withheld answer
# rather than raised.
#
# Two invariants are enforced on the released answer, and both are properties of
# the answer text itself rather than of the pipeline that produced it:
#   - no credential shape anywhere in it;
#   - the mandatory disclaimer is present. This template answers questions about
#     disclosure obligations; an answer released without the notice that it is not
#     legal advice is the one output shape this domain cannot ship.
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.services.service import detect_output_credentials
from src.services.llm_factory import resolve_llm
from src.services.llm_review import render_review, review_result

logger = logging.getLogger(__name__)

# Appended to every released answer.
DISCLAIMER = (
    "\n\n---\n"
    "【免責事項】本回答はAPPI 2026および少額短期保険の商品開示に関する"
    "一般的な情報提供を目的とし、法的助言を構成するものではありません。"
    "個別案件については資格を有する日本の法律・コンプライアンス専門家に"
    "ご確認ください。"
)

# The substring whose presence proves the disclaimer survived into the released
# answer. Checked on the assembled string, so a future renderer that drops it is
# caught by the boundary rather than by a reader.
DISCLAIMER_MARKER = "【免責事項】"

# Published when nothing was retrievable — a real answer, not a withholding.
NO_GROUNDING_ANSWER = (
    "十分な根拠情報が見つからなかったため、確定的な回答を提供できません。"
    "資格を有する専門家にご相談ください。" + DISCLAIMER
)

# Withheld-answer notice. Non-empty by contract: a falsy value here re-opens the
# envelope's fallback to the un-gated answer.
WITHHELD_NOTICE = (
    "回答は出力ポリシーにより保留されました。理由コード: {reason}。" "コンプライアンス窓口にお問い合わせください。"
)

# Closed set of withholding reasons. A reason is a label, never the matched value
# or the text that carried it.
REASON_CREDENTIAL = "credential_pattern_in_answer"
REASON_MISSING_DISCLAIMER = "mandatory_disclaimer_absent"
REASON_MALFORMED = "answer_channel_malformed"

# Every state field that carries answer text or a payload derived from it. On a
# withholding, all of them are overwritten — an inventory rather than a list of
# the ones that happened to be leaking, so a field added later has to be added
# here too or the boundary tests fail.
CLEARED_OUTPUT_FIELDS: tuple[str, ...] = ("result", "qa_answer", "citations")


def _cleared_output_state() -> Dict[str, Any]:
    """Return the delta that blanks every answer-bearing field.

    Each key is PRESENT with an empty value rather than omitted. LangGraph merges
    partial deltas, so omitting a key leaves the previous value in state — a
    "clearing" that omits is not a clearing at all, and an assertion written as
    `not result.get(field)` passes on it.
    """
    return {field: "" for field in CLEARED_OUTPUT_FIELDS}


class PostProcessNode(FunctionNode):
    """Gate the assembled answer and publish it, or withhold it.

    Outer backbone post_process slot. Declared ANONYMOUS: trust was enforced at
    PreProcessNode, and requiring more here would reject the context that boundary
    already cleared.

    Input state keys:
        result: str  — the assembled answer from the inner OutputFormatNode

    Output state keys (partial dict):
        formatted_output: str        — the released answer, or the withheld notice
        result / qa_answer / citations — blanked on a withholding
        status:           str
        error_log:        list[str]  (only on a withholding)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        raw_result = state.get("result")

        # A non-string answer channel is a withholding, not an exception. Raising
        # here would return a delta with no formatted_output, and the envelope
        # would fall back to the very value that could not be gated.
        if raw_result is not None and not isinstance(raw_result, str):
            return self._withhold(state, REASON_MALFORMED)

        result = raw_result or ""

        if not result.strip():
            logger.warning("PostProcessNode: no answer to release — publishing the no-grounding reply")
            emit_trace_event(
                "post_process_no_grounding",
                {"reason": "result_absent"},
                state,
            )
            return {
                "formatted_output": NO_GROUNDING_ANSWER,
                "status": AgentStatus.SUCCESS.value,
            }

        _llm, _ = resolve_llm(None, state)
        _remarks = review_result(
            _llm,
            user_input=str(state.get("user_input") or ""),
            result=result,
            domain="INS Microinsurance APPI 2026 & Disclosure Compliance Q&A Agent",
        )
        _review = render_review(_remarks)
        # Remarks are LLM text derived from the caller's raw words, so they pass through the
        # same gate the answer does -- appending after the gate would put unscanned text past
        # it. A tripped review is dropped on its own: withholding a correct answer because an
        # advisory remark quoted an identifier would let the review change the outcome, and
        # the whole design rests on it being unable to.
        if _review and isinstance(result, str) and not detect_output_credentials(result + _review + DISCLAIMER):
            result = result + _review

        formatted = result + DISCLAIMER

        violation = detect_output_credentials(formatted)
        if violation:
            return self._withhold(state, REASON_CREDENTIAL, detail=violation)

        if DISCLAIMER_MARKER not in formatted:
            return self._withhold(state, REASON_MISSING_DISCLAIMER)

        logger.info("PostProcessNode: released answer length=%d", len(formatted))
        emit_trace_event(
            "post_process_complete",
            {"released_length": len(formatted), "disclaimer_present": True},
            state,
        )

        return {
            "formatted_output": formatted,
            "status": AgentStatus.SUCCESS.value,
        }

    @staticmethod
    def _withhold(state: AgentState, reason: str, detail: str = "") -> Dict[str, Any]:
        """Withhold the answer: blank every answer-bearing field, publish a notice.

        The notice carries a reason code from the closed set above and nothing
        else. The detail — which pattern class fired — reaches the audit trail and
        the error log, never the caller-facing string, and it is a pattern NAME:
        the matched text is never recorded anywhere.
        """
        payload: Dict[str, Any] = {"reason": reason}
        if detail:
            payload["pattern"] = detail
        emit_trace_event("post_process_withheld", payload, state)
        logger.error("PostProcessNode: withheld the answer — %s", reason)

        errors: List[str] = [f"PostProcessNode: withheld the answer — {reason}"]
        if detail:
            errors = [f"PostProcessNode: withheld the answer — {reason} ({detail})"]

        return {
            **_cleared_output_state(),
            "formatted_output": WITHHELD_NOTICE.format(reason=reason),
            "status": AgentStatus.ERROR.value,
            "error_log": errors,
        }
