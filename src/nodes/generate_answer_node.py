"""AgentCore Platform v1.0"""

# INS-C2-038 — GenerateAnswerNode
#
# Inner domain node 4: compose the grounded compliance answer strictly from the
# passages that cleared the relevance floor, with a citation for each.
#
# The manifest declares generation_mode: deterministic — no model is invoked. The
# answer body is assembled only from retrieved passage text, so it cannot contain
# a claim no source carries. The declared grounding prompt
# (llm.system_prompt_template) states the contract a model-backed build must hold
# to; this node resolves the path and records whether it resolved, so a
# declaration naming a file that is not there is visible in the audit trail rather
# than silent.
#
# The question is quoted back into the answer, and that quote is the one place
# caller text enters a cited rendering. The answer marks each source with a
# leading marker character and a bracketed passage id, one per line — so a
# question containing a newline could otherwise open a line that reads exactly
# like a retrieved source and attribute invented text to a real regulator.
# neutralise_echo() collapses the quote to one line, removes the characters this
# renderer uses as structure, and bounds its length, so no caller string can be
# rendered as a source.
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

from src.schemas.state import from_json, to_json
from src.services.service import neutralise_echo, resolve_repo_path

logger = logging.getLogger(__name__)

# Returned when nothing cleared the relevance floor. Abstaining is the correct
# answer for a compliance question with no grounding: an unsourced statement about
# a disclosure obligation is worse than no statement.
UNGROUNDED_ANSWER = (
    "提示できる根拠資料が見つからなかったため、確定的な回答は控えます。"
    "ご質問をより具体的にしていただくか、資格を有する法律・コンプライアンス"
    "専門家にご相談ください。"
)

_NO_QUESTION_LABEL = "(質問未指定)"


def _synthesise_grounded_answer(
    quoted_question: str,
    passages: List[Dict[str, Any]],
) -> str:
    """Assemble the answer body from the grounded passages only.

    *quoted_question* has already been neutralised for rendering — see the module
    comment. Nothing outside the supplied passages is introduced.
    """
    lines: List[str] = [
        f"ご質問「{quoted_question}」について、以下の根拠資料に基づき回答します。",
        "",
        "【根拠情報に基づく回答】",
    ]
    for p in passages:
        lines.append(f"■ [{p.get('id')}] {p.get('title')}（出典: {p.get('source')}）")
        lines.append(f"  {p.get('text')}")
        lines.append("")
    lines.append(
        "【留意事項】上記は提示された根拠資料に基づく要約であり、確定的な法的助言"
        "ではありません。個別案件は資格を有する専門家にご確認ください。"
    )
    return "\n".join(lines)


class GenerateAnswerNode(FunctionNode):
    """Compose the grounded compliance answer.

    Inner node: ANONYMOUS trust (see the module comment).

    Input state keys:
        reranked_passages: str  — JSON-serialised passages that cleared the floor
        qa_query:          str  — JSON-serialised {question, normalized, topics}
        runtime_settings:  str  — JSON declared settings (system_prompt_template)

    Output state keys (partial dict):
        qa_answer: str  — the grounded answer text
        citations: str  — JSON-serialised list of {id, title, source}
        status:    str
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        passages: List[Dict[str, Any]] = from_json(state.get("reranked_passages"), []) or []
        qa_query: Dict[str, Any] = from_json(state.get("qa_query"), {}) or {}
        settings: Dict[str, Any] = from_json(state.get("runtime_settings"), {}) or {}

        quoted = neutralise_echo(qa_query.get("question", "")) or _NO_QUESTION_LABEL
        template_resolved = self._template_resolves(settings.get("system_prompt_template"))

        if not passages:
            logger.warning("GenerateAnswerNode: nothing cleared the relevance floor")
            emit_trace_event(
                "generate_answer_ungrounded",
                {
                    "reason": "no_grounded_passages",
                    "grounding_prompt_resolved": template_resolved,
                },
                state,
            )
            return {
                "qa_answer": UNGROUNDED_ANSWER,
                "citations": to_json([]),
                "status": AgentStatus.SUCCESS.value,
            }

        answer = _synthesise_grounded_answer(quoted, passages)
        citations = [{"id": p.get("id"), "title": p.get("title"), "source": p.get("source")} for p in passages]

        logger.info(
            "GenerateAnswerNode: grounded on %d passage(s) answer_len=%d",
            len(passages),
            len(answer),
        )
        emit_trace_event(
            "generate_answer_complete",
            {
                "grounded_passage_count": len(passages),
                "citation_ids": [c["id"] for c in citations],
                "answer_length": len(answer),
                "grounding_prompt_resolved": template_resolved,
            },
            state,
        )

        return {
            "qa_answer": answer,
            "citations": to_json(citations),
            "status": AgentStatus.SUCCESS.value,
        }

    @staticmethod
    def _template_resolves(declared: Any) -> bool:
        """True when the declared grounding prompt names a file inside the repository."""
        path = resolve_repo_path(declared)
        return bool(path and path.is_file())
