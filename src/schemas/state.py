"""AgentCore Platform v1.0"""

# State must be a flat TypedDict — never a Pydantic BaseModel. LangGraph
# checkpoints use msgpack serialization; Pydantic objects cause silent
# corruption. Extend AgentState with agent-specific fields only. Do NOT add
# credentials, secrets, or Pydantic models.
#
# INS-C2-038 — Microinsurance APPI 2026 & Product-Disclosure Compliance Q&A Agent
# Two-layer nested Cat 2 retrieval graph: outer backbone (AgentBaseGraph) + inner
# domain workflow (BaseGraph). The fields below cover both layers.
#
# All dict/list-valued fields are stored as JSON-serialized Optional[str]. Use
# to_json() / from_json() below at every producer and consumer node — one
# contract end to end. Never type a dict/list field as a bare dict/list; that
# causes msgpack serialization failures.

import json
from typing import Any, NotRequired, Optional

from framework.schemas.agent_state import AgentState


def to_json(value: Any) -> Optional[str]:
    """Serialize a value to a JSON string for State storage."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialize a JSON string from State storage."""
    if value is None:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class State(AgentState):
    """Flat TypedDict for INS-C2-038.

    All shared fields (user_input, status, session_id, node_history, error_log,
    hitl_*, etc.) are inherited from AgentState.

    formatted_output is NOT re-declared here — it is inherited from AgentState.
    """

    # ------------------------------------------------------------------
    # Outer layer — written by PreProcessNode (the trust and caller boundary)
    # ------------------------------------------------------------------

    # Screened, personal-data-masked, normalised compliance question.
    # Produced by PreProcessNode; consumed by the inner InputValidateNode.
    validated_input: NotRequired[Optional[str]]

    # JSON-serialised channel/request metadata. Shape: {"source": str, "channel": str}.
    enriched_context: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Declared runtime settings — seeded by DomainWorkflowGraph._extra_initial_state()
    # from config/config.yaml via the outer GraphNode. This is the ONLY route a
    # node can read a declared value from: the framework calls execute(state) with
    # one argument, so a node cannot receive the graph's config directly.
    # ------------------------------------------------------------------

    # JSON-serialised declared settings. Shape:
    #   {"top_k": int, "score_threshold": float, "hybrid_search": bool,
    #    "system_prompt_template": str}
    # Every value is bounds-checked before it is placed here; an invalid entry is
    # omitted and the consuming node falls back to its documented default.
    runtime_settings: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Inner layer — the retrieval domain nodes (DomainWorkflowGraph)
    # ------------------------------------------------------------------

    # JSON-serialised normalised query object.
    # Shape: {"question": str, "normalized": str, "topics": [str, ...]}
    qa_query: NotRequired[Optional[str]]

    # JSON-serialised list of retrieved candidate passages.
    # Each entry: {id, title, text, source, hits}
    retrieved_passages: NotRequired[Optional[str]]

    # JSON-serialised list of reranked + threshold-filtered passages.
    # Each entry: {id, title, text, source, score}
    reranked_passages: NotRequired[Optional[str]]

    # Number of grounded passages kept after rerank/filter.
    retrieved_count: NotRequired[Optional[int]]

    # Grounded answer text synthesised strictly from the reranked passages.
    qa_answer: NotRequired[Optional[str]]

    # JSON-serialised list of source citations. Each entry: {id, title, source}
    citations: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Outer layer — inner OutputFormatNode -> outer PostProcessNode
    # ------------------------------------------------------------------

    # Assembled answer + sources block, before the output boundary runs.
    # PostProcessNode gates it and writes the caller-facing formatted_output.
    result: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Tracing / audit — framework-managed; do NOT write from node code
    # ------------------------------------------------------------------

    trace_id: NotRequired[Optional[str]]
    correlation_id: NotRequired[Optional[str]]
    # node_history inherited from AgentState
