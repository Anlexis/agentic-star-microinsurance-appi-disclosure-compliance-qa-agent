"""AgentCore Platform v1.0"""

# INS-C2-038 — OutputFormatNode
#
# Inner domain node 5, and the last node of the inner graph. It appends the
# sources block to the grounded answer and writes the assembled string to
# `result`, which the outer PostProcessNode gates before anything reaches a
# caller.
#
# The sources block lists exactly the citations the answer was composed from, and
# each entry names a knowledge-base passage id. Nothing caller-supplied is
# rendered here.
#
# Inner node: ANONYMOUS trust (the trust boundary is PreProcessNode).
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import from_json
from src.services.source_disclosure import source_label

logger = logging.getLogger(__name__)

SOURCES_HEADING = "参考資料 / Sources:"


def _append_sources(answer: str, citations: List[Dict[str, Any]]) -> str:
    """Append the sources block to the grounded answer."""
    if not citations:
        return answer
    lines: List[str] = [answer, "", SOURCES_HEADING]
    for c in citations:
        lines.append(f"  - [{c.get('id')}] {c.get('title')}（{c.get('source')}）")
    return "\n".join(lines)


class OutputFormatNode(FunctionNode):
    """Assemble the answer and its sources block for the output boundary.

    Inner node: ANONYMOUS trust (see the module comment).

    Input state keys:
        qa_answer: str  — the grounded answer from GenerateAnswerNode
        citations: str  — JSON-serialised list of {id, title, source}

    Output state keys (partial dict):
        result:    str  — the assembled answer, for the outer output boundary
        qa_answer: str  — carried forward for observability
        status:    str
        error_log: list[str]  (only on failure)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        qa_answer = state.get("qa_answer")
        citations: List[Dict[str, Any]] = from_json(state.get("citations"), []) or []

        if not isinstance(qa_answer, str) or not qa_answer.strip():
            logger.error("OutputFormatNode: qa_answer is absent from state")
            emit_trace_event(
                "output_format_failed",
                {"reason": "qa_answer_absent", "field": "qa_answer"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["OutputFormatNode: qa_answer is absent from state"],
                # The runner surfaces `formatted_output or result` as `output`. A reason left only in
                # error_log reaches no one: the terminal result carries just `status`, and get_output()
                # does not copy error_log out of the graph -- the caller sees a blank spinner.
                "formatted_output": "Request could not be completed. "
                + ("OutputFormatNode: qa_answer is absent from state"),
            }

        result = _append_sources(qa_answer, citations)

        logger.info("OutputFormatNode: result_len=%d citations=%d", len(result), len(citations))
        # Say where the answer came from. This agent answers from a corpus defined inside
        # its own module; a reader seeing a citation has no way to tell that from a live query
        # against the system of record, and the review round rated that confusion its most
        # serious finding. Added before the gate below so it passes the same checks the answer
        # does.
        result = result + source_label(state)
        emit_trace_event(
            "output_format_complete",
            {"result_length": len(result), "citation_count": len(citations)},
            state,
        )

        return {
            "result": result,
            "qa_answer": qa_answer,
            "status": AgentStatus.SUCCESS.value,
        }
