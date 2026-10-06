"""AgentCore Platform v1.0"""

# INS-C2-038 — outer graph (AgentBaseGraph; two-layer nested retrieval architecture)
#
# Architecture:
#
#   Outer backbone (fixed — do NOT override add_edges()):
#     START -> initialize -> pre_process -> main -> {route} -> post_process
#              -> finalize -> END
#                                             |
#                                             +-- (RETRY) -> pre_process
#
#   The `main` slot is a GraphNode subclass (ComplianceQaGraphNode) that delegates
#   the whole retrieval workflow to DomainWorkflowGraph (inner BaseGraph):
#     input_validate -> retrieve -> rerank_filter -> generate_answer -> output_format
#
#   Domain complexity is fully encapsulated inside the inner graph; the outer
#   backbone is never modified.
#
# Directory layout:
#   src/graph/graph.py                 <- outer graph (this file)
#   src/graph/domain_workflow_graph.py <- inner graph (retrieval topology)
#
# Runtime settings: config/config.yaml is the only declaration file the running
# agent reads. The manifest (config/agent.yaml) is flat and carries no runtime
# block, so a reader pointed at it would silently return an empty mapping and
# every declared value would go dead while the suite stayed green. _parent_config()
# reads config/config.yaml, bounds-checks each value, and hands the survivors to
# the inner graph, which seeds them into state — the only route a node can read
# them from, because the framework calls execute(state) with one argument.

import os
from typing import Any, ClassVar, Dict, Optional, Tuple

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.trust_level import TrustLevel
from framework.schemas.agent_state import AgentState
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State
from src.services.service import detect_output_credentials, finite_in_range

# Runtime parameters live at config/config.yaml, three levels up from this file
# (src/graph/graph.py -> src/graph -> src -> repository root).
_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "config.yaml",
)

# Bounds for each declared retrieval setting. A value outside its range, or one
# that is not a real finite number, is dropped rather than clamped: a caller of
# this template should see its own declaration refused, not silently rewritten.
_TOP_K_BOUNDS: Tuple[float, float] = (1, 64)
_SCORE_THRESHOLD_BOUNDS: Tuple[float, float] = (0.0, 1.0)


def _load_runtime_config() -> Dict[str, Any]:
    """Return the runtime parameters declared in config/config.yaml.

    Best-effort: a missing or unparseable file yields an empty mapping so graph
    construction never breaks, and every consuming node then falls back to its
    documented default. PyYAML is imported on demand — it is a framework runtime
    dependency, so loading it lazily avoids a hard module-load coupling.
    """
    try:
        import yaml

        with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


class ComplianceQaGraphNode(GraphNode):
    """The `main` slot of MicroinsuranceAppiComplianceQaAgent.

    Wraps DomainWorkflowGraph, the inner retrieval graph. Called by the backbone
    after pre_process and before post_process.

    Contracts:
      get_subgraph()  — build DomainWorkflowGraph with the declared settings
      extract_input() — take the screened question written by PreProcessNode
      merge_output()  — map the inner result into the outer state delta
      error_strategy  — "propagate": an inner failure surfaces as a failure here
    """

    # S-1 declared on the wrapper too: the CI gate only AST-scans FunctionNode
    # subclasses, so a GraphNode main slot passes the pipeline without one and is
    # flagged at review. Same level the nodes in this repo already declare.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    error_strategy: ClassVar[str] = "propagate"
    propagate_hitl: ClassVar[bool] = False

    def get_subgraph(self) -> "Any":
        """Build the inner retrieval workflow graph with the declared settings.

        DomainWorkflowGraph is imported inside the method to avoid a circular
        import at module load.
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._parent_config())

    def extract_input(self, state: AgentState) -> str:
        """Return the string handed to the inner graph's invoke().

        PreProcessNode owns the caller contract: it screens the raw question,
        masks personal data, and writes the survivor to validated_input. Falling
        back to the raw user_input would hand the inner graph a string that never
        passed those checks, so the fallback is an empty string — the inner graph
        then takes its documented no-question path.
        """
        validated = state.get("validated_input")
        return validated if isinstance(validated, str) else ""

    def merge_output(self, state: AgentState, sub_result: Dict[str, Any]) -> Dict[str, Any]:
        """Map the inner result into the outer state delta (changed keys only).

        `result` is the assembled answer, and it is what the base envelope falls
        back to when no formatted_output is published — so whatever lands here
        must be something the output boundary can gate. This method deliberately
        does NOT screen its shape: PostProcessNode already handles an answer
        channel it cannot read, and a second guard over the same fault would mean
        removing either one leaves the boundary test green. The containment sits
        where the leak surfaces, and nowhere else, so it stays falsifiable.
        """
        return {
            "qa_answer": sub_result.get("qa_answer"),
            "citations": sub_result.get("citations"),
            "retrieved_count": sub_result.get("retrieved_count", 0),
            "result": sub_result.get("result"),
            "status": sub_result.get("status"),
        }

    def _parent_config(self) -> Dict[str, Any]:
        """Bounds-check the declared runtime settings and forward the survivors.

        Only values the declaration actually carries, and that survive their
        bounds check, are forwarded; anything else is omitted so the consuming
        node falls back to its own documented default. The mapping is placed under
        the LangGraph ``configurable`` key, which DomainWorkflowGraph reads in
        _extra_initial_state() to seed the inner state.
        """
        cfg = _load_runtime_config()
        retrieval = cfg.get("retrieval") or {}
        llm = cfg.get("llm") or {}
        declared: Dict[str, Any] = {}

        top_k = finite_in_range(retrieval.get("top_k"), *_TOP_K_BOUNDS)
        if top_k is not None:
            declared["top_k"] = int(top_k)

        threshold = finite_in_range(retrieval.get("score_threshold"), *_SCORE_THRESHOLD_BOUNDS)
        if threshold is not None:
            declared["score_threshold"] = threshold

        hybrid = retrieval.get("hybrid_search")
        if isinstance(hybrid, bool):
            declared["hybrid_search"] = hybrid

        template = llm.get("system_prompt_template")
        if isinstance(template, str) and template:
            declared["system_prompt_template"] = template

        return {"configurable": declared}


class MicroinsuranceAppiComplianceQaAgent(AgentBaseGraph):
    """Outer graph for INS-C2-038.

    Inherits AgentBaseGraph directly. Domain logic is fully encapsulated in
    ComplianceQaGraphNode (main slot), which delegates to DomainWorkflowGraph.

    Backbone:
        START -> initialize -> pre_process -> main -> post_process -> finalize -> END

    register_nodes() is the only override:
      - super().register_nodes() fills initialize and finalize
      - pre_process:  PreProcessNode  (VERIFIED_EXTERNAL — the trust boundary)
      - main:         ComplianceQaGraphNode
      - post_process: PostProcessNode (the output boundary)

    add_edges() is NOT overridden — backbone wiring belongs to the framework.
    The class name must match the config/agent.yaml `class:` entry exactly.
    """

    @property
    def name(self) -> str:
        """Agent identifier registered with the agent registry."""
        return "MicroinsuranceAppiComplianceQaAgent"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill all five backbone slots.

        super().register_nodes() must be called first — it injects the framework's
        default initialize node (schema_version, session_id, trust_level) and
        finalize node (response metadata, total time).
        """
        super().register_nodes()  # fills: initialize, finalize

        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = ComplianceQaGraphNode()
        self._nodes["post_process"] = PostProcessNode()

    # get_output() is deliberately NOT overridden.
    #
    # The whole caller-facing answer — the grounded text, the sources block and
    # the mandatory disclaimer — is one string, which PostProcessNode writes to
    # formatted_output; the base envelope surfaces it as `output`. The structured
    # keys (qa_answer / citations / retrieved_count) are secondary metadata
    # already contained in that string, so an override would add no caller value.
    #
    # It would also cost assurance. The base envelope resolves
    # `formatted_output or result`, so an override blanking `output` on a
    # non-success status would contain the output boundary's leak on its own —
    # and thereby make the boundary's own clearing unfalsifiable. The clearing is
    # where the containment belongs, so it is the only layer holding it, and
    # removing it is what the boundary tests detect.

    # ── Output boundary, agent-class layer ────────────────────────────────────
    def _security_gate_output(self, content: str) -> Optional[str]:
        """Return the name of the first credential shape in the final answer.

        Defence in depth alongside the scan PostProcessNode performs. Both call
        the same detector so the two layers cannot drift into different refusal
        sets — a narrower one would pass a value the framework then raises on,
        discarding the node's containment with the delta.
        """
        if not content:
            return None
        return detect_output_credentials(content)


# Alias kept for the standalone entry point, which imports `Graph`.
Graph = MicroinsuranceAppiComplianceQaAgent
