"""AgentCore Platform v1.0"""

# INS-C2-038 — PreProcessNode
#
# The outer backbone's pre_process slot, and the template's caller boundary. It
# is the only node that requires VERIFIED_EXTERNAL trust, so an unauthenticated
# caller is refused here and the inner retrieval nodes never see untrusted text.
#
# Everything the template guarantees about caller input is enforced in this node,
# not delegated to the framework's own input policy. The framework's policy runs
# in front of every node and does catch several instruction-override forms — but
# it scores the `<<SYS>>` system-block marker at nothing, and a template whose
# only screen is the framework's is fail-OPEN wherever that policy is absent or
# configured off. The checks below are proved by calling execute() directly, with
# no framework wrapper in front.
#
# Order matters. The question is screened first, then masked: screening a string
# that masking has already rewritten would be screening a different string from
# the one the caller sent.
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import to_json
from src.services.service import is_inert_token, mask_personal_data, screen_text

logger = logging.getLogger(__name__)

# Upper bound on the accepted question. A compliance question that needs more
# than this is not a question; the cap also bounds the work every downstream node
# does on caller-controlled text.
MAX_QUESTION_CHARS = 4000

# Refusal reasons are a closed set. A refusal names the reason and the field, and
# never the value that caused it — an error message is an output channel, and a
# rejected string echoed into one is the same disclosure the check exists to
# prevent.
REASON_EMPTY = "question_missing_or_empty"
REASON_TOO_LONG = "question_exceeds_length_limit"
REASON_SCREENED = "question_carries_instruction_override"


class PreProcessNode(FunctionNode):
    """Trust boundary and caller contract for INS-C2-038.

    Input state keys:
        user_input:     str   — the caller's compliance question, plain text
        input_context:  dict  — read-only channel metadata

    Output state keys (partial dict):
        validated_input:  str        — screened, masked, normalised question
        enriched_context: str        — JSON-serialised channel metadata
        status:           str
        error_log:        list[str]  — set only on refusal
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: AgentState) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        input_context = state.get("input_context") or {}

        if not isinstance(user_input, str) or not user_input.strip():
            return self._refuse(state, REASON_EMPTY, "user_input")

        stripped = user_input.strip()

        if len(stripped) > MAX_QUESTION_CHARS:
            emit_trace_event(
                "pre_process_rejected",
                {"reason": REASON_TOO_LONG, "field": "user_input", "length": len(stripped)},
                state,
            )
            logger.warning("PreProcessNode: refused user_input — %s", REASON_TOO_LONG)
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [
                    f"PreProcessNode: refused user_input — {REASON_TOO_LONG} "
                    f"(limit {MAX_QUESTION_CHARS} characters)"
                ],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + (f"PreProcessNode: refused user_input — {REASON_TOO_LONG} (limit {MAX_QUESTION_CHARS} characters)"),
            }

        # Instruction-override screening, on the question as received.
        screened = screen_text(stripped)
        if screened:
            return self._refuse(state, REASON_SCREENED, "user_input", detail=screened)

        # Personal-data masking. The framework's own masker anchors on word
        # boundaries, which do not exist between a Kanji and a digit, so a value
        # written the way a Japanese policyholder writes it survives it. This pass
        # uses explicit character guards and does not depend on the script.
        masked, kinds = mask_personal_data(stripped)

        channel = input_context.get("channel") if isinstance(input_context, dict) else None
        channel_label = channel if is_inert_token(channel) else "unknown"

        logger.info(
            "PreProcessNode: accepted question length=%d masked_kinds=%d",
            len(masked),
            len(kinds),
        )
        emit_trace_event(
            "pre_process_validated",
            {
                "question_length": len(masked),
                "personal_data_kinds": kinds,
                "channel": channel_label,
            },
            state,
        )

        return {
            "validated_input": masked,
            "enriched_context": to_json({"source": "MicroinsuranceAppiComplianceQaAgent", "channel": channel_label}),
            "status": AgentStatus.SUCCESS.value,
        }

    @staticmethod
    def _refuse(
        state: AgentState,
        reason: str,
        field: str,
        detail: str = "",
    ) -> Dict[str, Any]:
        """Refuse the request, naming the reason and the field but never the value."""
        payload: Dict[str, Any] = {"reason": reason, "field": field}
        if detail:
            payload["pattern"] = detail
        emit_trace_event("pre_process_rejected", payload, state)
        logger.warning("PreProcessNode: refused %s — %s", field, reason)
        message = f"PreProcessNode: refused {field} — {reason}"
        if detail:
            message = f"{message} ({detail})"
        errors: List[str] = [message]
        _error_lines = errors
        # The runner surfaces `formatted_output or result` as `output`. A reason left only in
        # error_log reaches no one: the terminal result carries just `status`, and get_output()
        # does not copy error_log out of the graph -- the caller sees a blank spinner.
        # The list is bound once: repeating the expression inline would evaluate it twice.
        return {
            "status": AgentStatus.ERROR.value,
            "error_log": _error_lines,
            # A SCREENED refusal stays silent: its message names the marker that caught the
            # payload, so returning it lets an attacker probe the screen one try at a time.
            # A VALIDATION refusal names the rule, which is what the caller needs to fix it.
            **(
                {}
                if reason == REASON_SCREENED
                else {
                    "formatted_output": "Request could not be completed. "
                    + "; ".join(str(_line) for _line in _error_lines)
                }
            ),
        }
