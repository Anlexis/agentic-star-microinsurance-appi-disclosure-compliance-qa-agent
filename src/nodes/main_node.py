"""AgentCore Platform v1.0"""

# INS-C2-038 — MainNode
#
# The `main` slot of MicroinsuranceAppiComplianceQaAgent is ComplianceQaGraphNode
# (src/graph/graph.py), which delegates the retrieval workflow to
# DomainWorkflowGraph. MainNode is NOT that slot: it is the reference
# implementation of the simpler single-node `main` shape used by
# src/examples/graph_cat1_sample.py, kept alongside the composed build so the
# simple shape has a live, exercised example next to the nested one.
#
# Its contract is intentional and pinned by tests:
#   - override execute(self, state: AgentState) -> dict
#   - return the partial dict {status, result}
#   - emit exactly one domain audit event

from typing import Any, ClassVar, Dict

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

MAIN_NODE_REFERENCE_MSG = (
    "MainNode (single-slot reference): the production main slot of this template "
    "is the ComplianceQaGraphNode -> DomainWorkflowGraph pipeline. This node is "
    "the single-slot reference implementation used by src/examples/."
)


class MainNode(FunctionNode):
    """Single-slot `main`-node reference implementation.

    Not the production main slot (that is ComplianceQaGraphNode in graph.py).
    Backs src/examples/graph_cat1_sample.py and is pinned by
    tests/unit/test_main_node.py.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        emit_trace_event(
            "main_node_reference_called",
            {
                "role": "single-slot reference (not the composed production main slot)",
                "production_main": "ComplianceQaGraphNode + DomainWorkflowGraph",
            },
            state,
        )
        return {
            "status": AgentStatus.SUCCESS.value,
            "result": MAIN_NODE_REFERENCE_MSG,
        }
