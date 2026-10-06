"""AgentCore Platform v1.0"""

# INS-C2-038 — RerankFilterNode
#
# Inner domain node 3: rerank the retrieved passages and drop those below the
# declared relevance floor, so answer generation is grounded only on sources that
# actually matched.
#
# Scoring normalises each passage's hit count against the best-scoring passage, so
# score_threshold is read on a [0, 1] scale regardless of how many terms a
# question happened to carry.
#
# One declared setting is read here, from state:
#   score_threshold — the relevance floor, on that normalised scale
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
from src.services.service import finite_in_range

logger = logging.getLogger(__name__)

# Used when score_threshold is absent from the declared settings or failed its
# bounds check.
DEFAULT_SCORE_THRESHOLD = 0.75
_THRESHOLD_BOUNDS = (0.0, 1.0)


def _rerank_and_filter(
    passages: List[Dict[str, Any]],
    threshold: float,
) -> List[Dict[str, Any]]:
    """Normalise hits to a [0, 1] score and keep those at or above *threshold*.

    Never mutates the input list — builds a fresh one.
    """
    if not passages:
        return []
    max_hits = max(int(p.get("hits", 0)) for p in passages) or 1
    ranked: List[Dict[str, Any]] = []
    for p in passages:
        score = round(int(p.get("hits", 0)) / max_hits, 4)
        ranked.append(
            {
                "id": p.get("id"),
                "title": p.get("title"),
                "text": p.get("text"),
                "source": p.get("source"),
                "score": score,
            }
        )
    ranked.sort(key=lambda p: p["score"], reverse=True)
    return [p for p in ranked if p["score"] >= threshold]


class RerankFilterNode(FunctionNode):
    """Rerank retrieved passages and apply the declared relevance floor.

    Inner node: ANONYMOUS trust (see the module comment).

    Input state keys:
        retrieved_passages: str  — JSON-serialised list from RetrieveNode
        runtime_settings:   str  — JSON declared settings (score_threshold)

    Output state keys (partial dict):
        reranked_passages: str  — JSON-serialised passages that cleared the floor
        retrieved_count:   int  — how many cleared it
        status:            str
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        passages: List[Dict[str, Any]] = from_json(state.get("retrieved_passages"), []) or []

        settings: Dict[str, Any] = from_json(state.get("runtime_settings"), {}) or {}
        declared = finite_in_range(settings.get("score_threshold"), *_THRESHOLD_BOUNDS)
        threshold = declared if declared is not None else DEFAULT_SCORE_THRESHOLD

        if not passages:
            logger.warning("RerankFilterNode: nothing retrieved to rerank")
            emit_trace_event(
                "rerank_filter_empty",
                {"reason": "no_retrieved_passages"},
                state,
            )
            return {
                "reranked_passages": to_json([]),
                "retrieved_count": 0,
                "status": AgentStatus.SUCCESS.value,
            }

        kept = _rerank_and_filter(passages, threshold)

        logger.info(
            "RerankFilterNode: in=%d threshold=%.4f kept=%d",
            len(passages),
            threshold,
            len(kept),
        )
        emit_trace_event(
            "rerank_filter_complete",
            {
                "score_threshold": threshold,
                "input_count": len(passages),
                "kept_count": len(kept),
                "kept_ids": [p["id"] for p in kept],
            },
            state,
        )

        return {
            "reranked_passages": to_json(kept),
            "retrieved_count": len(kept),
            "status": AgentStatus.SUCCESS.value,
        }
