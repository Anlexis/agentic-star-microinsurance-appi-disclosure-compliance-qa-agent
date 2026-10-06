"""AgentCore Platform v1.0"""

# INS-C2-038 — InputValidateNode
#
# Inner domain node 1: domain-level normalisation of the compliance question and
# topic classification (APPI 2026 underwriting-AI data use vs 少額短期保険 product
# disclosure). The topics steer nothing on their own — they are recorded so an
# operator can see which body of rules a question was read against.
#
# Distinct from PreProcessNode, which owns the trust boundary and the caller
# contract. By the time this node runs, the question has already been screened
# and masked; what is left here is domain shaping.
#
# Inner node: ANONYMOUS trust (the trust boundary is PreProcessNode; an inner node
# demanding more would reject the context that boundary already cleared).
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import re
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import to_json

logger = logging.getLogger(__name__)

# Topic-detection keyword sets (read-only; never mutated in execute()).
_APPI_KEYWORDS: List[str] = [
    "appi",
    "個人情報",
    "個人データ",
    "consent",
    "同意",
    "利用目的",
    "purpose",
    "underwriting",
    "引受",
    "pricing",
    "profiling",
    "プロファイリング",
    "automated",
    "自動化",
    "要配慮",
    "sensitive",
    "third party",
    "第三者提供",
]
_SSTI_KEYWORDS: List[str] = [
    "少額短期",
    "ssti",
    "disclosure",
    "開示",
    "prospectus",
    "契約概要",
    "注意喚起",
    "重要事項",
    "warning",
    "警告",
    "cooling",
    "クーリングオフ",
    "電子交付",
    "digital",
    "電子",
    "登録",
    "監督指針",
    "保険業法",
]

_WHITESPACE_RE = re.compile(r"\s+")


def _detect_topics(text: str) -> List[str]:
    """Return the topic tags a normalised question carries."""
    low = text.lower()
    topics: List[str] = []
    if any(kw.lower() in low for kw in _APPI_KEYWORDS):
        topics.append("appi_2026")
    if any(kw.lower() in low for kw in _SSTI_KEYWORDS):
        topics.append("ssti_disclosure")
    if not topics:
        topics.append("general_compliance")
    return topics


class InputValidateNode(FunctionNode):
    """Normalise the compliance question and tag the rules it falls under.

    Inner node: ANONYMOUS trust (see the module comment).

    Input state keys:
        user_input: str  — the inner graph's own input channel. The inner graph is
                           invoked with the string ComplianceQaGraphNode.extract_input()
                           returned, which is the screened, masked question
                           PreProcessNode wrote to the OUTER state's validated_input.
                           The inner state is built fresh by the framework, so the
                           outer channel name does not exist here.

    Output state keys (partial dict):
        qa_query:  str  — JSON-serialised {question, normalized, topics}
        status:    str
        error_log: list[str]  (only on failure)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        raw = state.get("user_input")

        if not isinstance(raw, str) or not raw.strip():
            logger.error("InputValidateNode: no question arrived on the inner input channel")
            emit_trace_event(
                "input_validate_failed",
                {"reason": "question_absent", "field": "user_input"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["InputValidateNode: no question arrived on the inner input channel"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("InputValidateNode: no question arrived on the inner input channel"),
            }

        # One representation, not two. The question and its normalised form used
        # to be stored separately, and the renderer quoted the un-normalised one —
        # so a newline the caller wrote survived into a cited answer. Collapsing
        # here means every consumer sees the same single-line string.
        normalized = _WHITESPACE_RE.sub(" ", raw.strip())
        topics = _detect_topics(normalized)

        qa_query: Dict[str, Any] = {
            "question": normalized,
            "normalized": normalized,
            "topics": topics,
        }

        logger.info("InputValidateNode: question_len=%d topics=%s", len(normalized), topics)
        emit_trace_event(
            "input_validate_complete",
            {"question_length": len(normalized), "topics": topics},
            state,
        )

        return {
            "qa_query": to_json(qa_query),
            "status": AgentStatus.SUCCESS.value,
        }
