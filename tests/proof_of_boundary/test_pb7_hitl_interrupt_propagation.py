# PB-7: HITL Interrupt-Propagation Boundary Test
#
# PB-7 verifies that a human-in-the-loop (HITL) interrupt raised inside the
# agent graph propagates ACROSS the backbone / subgraph boundary up to the
# invoking caller — so an external orchestrator can pause the run, collect a
# human decision, and resume it. This behaviour is only meaningful for
# templates that opt into cross-boundary HITL propagation: a main-slot
# GraphNode declaring `propagate_hitl = True`, or an explicit interrupt()
# checkpoint on the 5-node backbone.
#
# INS-C2-038 (Microinsurance APPI 2026 & Disclosure Compliance Q&A) does NOT
# enable cross-boundary HITL interrupt propagation: ComplianceQaGraphNode
# declares `propagate_hitl = False`, the inner DomainWorkflowGraph runs a fully
# linear, rule-based RAG pipeline with no interrupt() checkpoint, and the
# backbone runs to completion. There is therefore no propagation behaviour to
# assert, so PB-7 ships as a skip stub — a real,
# importable module that skips with a clear reason (never `assert True`), ready
# to be filled in if/when HITL propagation is wired end-to-end.

import importlib

import pytest


def _hitl_propagation_enabled() -> bool:
    """True iff this template opts into cross-boundary HITL interrupt propagation.

    Detected by inspecting the classes DEFINED in ``src/graph/graph.py`` for a
    node/graph subclass that declares ``propagate_hitl = True``. Imported
    defensively so PB-7 collection never errors when the SDK wheel or the graph
    module is unavailable — PB-7 then simply skips.
    """
    try:
        graph_mod = importlib.import_module("src.graph.graph")
    except Exception:
        return False
    for obj in vars(graph_mod).values():
        if (
            isinstance(obj, type)
            and getattr(obj, "__module__", None) == graph_mod.__name__
            and getattr(obj, "propagate_hitl", False) is True
        ):
            return True
    return False


_HITL_PROPAGATION_ENABLED = _hitl_propagation_enabled()

_PB7_SKIP_REASON = (
    "HITL interrupt-propagation not implemented for this template "
    "(ComplianceQaGraphNode.propagate_hitl=False; no cross-boundary "
    "interrupt() checkpoint) — PB-7 skip stub"
)


@pytest.mark.skipif(not _HITL_PROPAGATION_ENABLED, reason=_PB7_SKIP_REASON)
class TestPB7HitlInterruptPropagation:
    """PB-7: a HITL interrupt must propagate across the graph boundary.

    Skipped for this template — cross-boundary HITL propagation is not enabled,
    so there is no interrupt-propagation behaviour to verify. The real
    assertion is implemented here once HITL propagation is wired end-to-end.
    """

    def test_hitl_interrupt_propagates_to_caller(self):
        # Reached only when a graph class declares propagate_hitl=True.
        # The real PB-7 assertion (invoke -> assert the interrupt surfaces to the
        # caller -> resume) is implemented at that point.
        raise AssertionError("PB-7 real assertion not yet implemented for a HITL-enabled template")
