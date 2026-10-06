# INS-C2-038 — Unit Tests: Main Node

from src.nodes.main_node import MainNode
from src.nodes.pre_process_node import PreProcessNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel


class TestMainNode:
    """Unit tests for the main business logic node."""

    def setup_method(self):
        self.node = MainNode()

    def test_success_path(self):
        """TC: Main node processes valid input and returns SUCCESS.

        Invoked via __call__ (node(state)) — not .execute() directly — so the
        BaseNode S-1 trust gate runs first. MainNode is ANONYMOUS,
        so an ANONYMOUS caller is admitted and execute() runs to completion.
        """
        state = {
            "validated_input": "test input",
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
            "node_history": [],
            "error_log": [],
        }
        result = self.node(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["result"] is not None

    def test_empty_input(self):
        """TC: Main node handles empty input gracefully.

        Routed through __call__ (S-1 gate first); ANONYMOUS caller admitted.
        """
        state = {
            "validated_input": "",
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
            "node_history": [],
            "error_log": [],
        }
        result = self.node(state)
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_execute_method_signature(self):
        """Node contract: Node must implement execute(state) not _invoke_impl.

        Canonical node contract:
          - Override: execute(self, state: AgentState) -> dict
          - PROHIBITED: _invoke_impl(), process() override
        """
        import inspect

        # Must have execute() defined on the concrete class (not just inherited stub)
        assert hasattr(MainNode, "execute"), "MainNode must implement execute()"

        sig = inspect.signature(MainNode.execute)
        params = list(sig.parameters.keys())
        # execute(self, state) — at minimum two parameters
        assert len(params) >= 2, f"execute() must accept (self, state), got params: {params}"
        assert params[1] == "state", f"Second parameter must be 'state', got '{params[1]}'"

        # Must NOT define _invoke_impl at the domain level
        assert (
            "_invoke_impl" not in MainNode.__dict__
        ), "_invoke_impl() must not be defined in MainNode — use execute() instead"


class TestS1TrustGate:
    """S-1 trust-gate coverage.

    Prior unit tests invoked nodes via ``node.execute(state)``, which bypasses
    ``BaseNode.__call__`` — the security boundary where the S-1 trust gate runs
    BEFORE ``execute()``.  These tests invoke through ``node(state)`` so the gate
    is actually exercised, and add explicit negative + positive authorization
    coverage.

    Target node: **PreProcessNode** — the outer backbone's ``pre_process`` slot
    and the only node that requires ``VERIFIED_EXTERNAL`` trust (every other
    node in the template is ``ANONYMOUS``).  The S-1 gate RETURNS an error dict
    on insufficient trust (it does not raise), so we assert on the returned dict.
    """

    # PII-free compliance question: lowercase, no '@', no digit groups, no two
    # consecutive Title-Case words — so the S-2 input gate never masks it and
    # the positive-control path reaches execute() unaltered.
    PAYLOAD = "what does my microinsurance policy cover for flood damage"

    def setup_method(self):
        self.node = PreProcessNode()

    def test_s1_gate_rejects_untrusted_caller_before_execute(self):
        """Negative auth: an ANONYMOUS caller is denied by the S-1 gate.

        PreProcessNode requires VERIFIED_EXTERNAL. Invoked via __call__, the gate
        returns an ERROR dict, names the trust-gate denial in error_log, and
        execute() never runs (so no ``validated_input`` is produced).
        """
        state = {
            "user_input": self.PAYLOAD,
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
            "node_history": [],
            "error_log": [],
        }
        result = self.node(state)  # via BaseNode.__call__ — S-1 gate runs first

        assert result["status"] == AgentStatus.ERROR.value
        assert "trust gate denied" in " ".join(result.get("error_log", []))
        # execute() never ran → the gate short-circuited before any output
        assert "validated_input" not in result

    def test_s1_gate_admits_trusted_caller(self):
        """Positive control: a VERIFIED_EXTERNAL caller passes the S-1 gate and
        execute() runs to completion (SUCCESS), proving the rejection above is
        the gate — not an unrelated failure.
        """
        state = {
            "user_input": self.PAYLOAD,
            "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
            "input_context": {"channel": "web"},
            "node_history": [],
            "error_log": [],
        }
        result = self.node(state)  # via BaseNode.__call__ — S-1 gate admits

        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["validated_input"] == self.PAYLOAD
