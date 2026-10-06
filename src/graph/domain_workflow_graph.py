"""AgentCore Platform v1.0"""

# INS-C2-038 — DomainWorkflowGraph (inner BaseGraph)
#
# The inner graph of the two-layer nested architecture. It holds the whole
# compliance retrieval pipeline:
#
#   START
#     -> input_validate   (InputValidateNode)
#     -> retrieve         (RetrieveNode)
#     -> rerank_filter    (RerankFilterNode)
#     -> generate_answer  (GenerateAnswerNode)
#     -> output_format    (OutputFormatNode)
#     -> END
#
# Called by ComplianceQaGraphNode.get_subgraph() in graph.py; get_output() shapes
# the dict that node's merge_output() consumes.
#
# Rules this file holds to:
#   - inherits BaseGraph (fully custom topology, no forced backbone)
#   - register_nodes() does not call super() (it is abstract in BaseGraph)
#   - initialize / finalize are outer-backbone concerns and are not registered
#   - every inner node declares ANONYMOUS trust: the trust boundary is the outer
#     PreProcessNode, and an inner node demanding more would reject the context
#     that boundary already cleared
#   - every inner node is constructed with no arguments

from typing import Any, Dict

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State, to_json


class DomainWorkflowGraph(BaseGraph):
    """Inner retrieval workflow graph for INS-C2-038.

    Pipeline (linear):
        START -> input_validate -> retrieve -> rerank_filter -> generate_answer
              -> output_format -> END
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "ins_c2_038_appi_compliance_qa_workflow"

    @property
    def state_schema(self) -> type:
        return State

    # ── Config validation ─────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        """No mandatory configuration.

        Every declared retrieval setting is optional and was already bounds-
        checked by the outer graph before it arrived here; an absent value means
        the consuming node uses its own documented default.
        """

    # ── Runtime settings seeding ──────────────────────────────────────────────

    def _extra_initial_state(self) -> Dict[str, Any]:
        """Seed the declared runtime settings into the inner initial state.

        This is the only route a declared value can reach a node. The framework
        calls ``node.execute(state)`` with a single argument, so a node cannot be
        handed the graph's config directly; anything a node must read has to be a
        state channel. The outer graph bounds-checks each declared value and puts
        the survivors under ``configurable``; they are serialised into one
        ``runtime_settings`` channel here.
        """
        configurable: Dict[str, Any] = (self.config or {}).get("configurable") or {}
        if not configurable:
            return {}
        return {"runtime_settings": to_json(dict(configurable))}

    # ── Node registration ─────────────────────────────────────────────────────

    def register_nodes(self) -> None:
        """Register the five retrieval domain nodes."""
        self._nodes["input_validate"] = InputValidateNode()
        self._nodes["retrieve"] = RetrieveNode()
        self._nodes["rerank_filter"] = RerankFilterNode()
        self._nodes["generate_answer"] = GenerateAnswerNode()
        self._nodes["output_format"] = OutputFormatNode()

    # ── Edge wiring ───────────────────────────────────────────────────────────

    def add_edges(self) -> None:
        """Wire the linear retrieval topology."""
        self._sg.add_edge(START, "input_validate")
        self._sg.add_edge("input_validate", "retrieve")
        self._sg.add_edge("retrieve", "rerank_filter")
        self._sg.add_edge("rerank_filter", "generate_answer")
        self._sg.add_edge("generate_answer", "output_format")
        self._sg.add_edge("output_format", END)

    # ── Routing ───────────────────────────────────────────────────────────────

    def route(self, state: AgentState) -> str:
        """Required by the BaseGraph contract; not used by this topology.

        The pipeline is linear, so add_conditional_edges() is never called and
        this method is not reached at runtime. It returns END on a failed state so
        an unexpected invocation cannot re-enter a processing node.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return END
        return "output_format"

    # ── Output shape ──────────────────────────────────────────────────────────

    def get_output(self, state: AgentState) -> Dict[str, Any]:
        """Shape the dict returned to the outer graph as sub_result.

        Every value is read straight from state and no key is resolved with `or`:
        a fallback chain here would recreate, one level down, the same
        "first truthy wins" behaviour that lets a gated-empty field re-open an
        un-gated one. ComplianceQaGraphNode.merge_output() reads exactly these
        keys.
        """
        return {
            "qa_answer": state.get("qa_answer"),
            "citations": state.get("citations"),
            "retrieved_count": state.get("retrieved_count", 0),
            "result": state.get("result"),
            "status": state.get("status"),
            "node_history": state.get("node_history", []),
            "correlation_id": state.get("correlation_id"),
        }
