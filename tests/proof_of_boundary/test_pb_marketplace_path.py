"""The Marketplace entry path: the graph, invoked the way the Pod invokes it.

The Pod runs `cli.py -> run_agent_marketplace()`, which always seeds
`input_context={"conversation_history": history}` and never goes through
`src/api/server.py`. A filter that lives in the HTTP adapter is invisible here --
which is how a whole fleet refused every chat message while its tests stayed green.
"""

import ast
import importlib
from pathlib import Path

import pytest

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.utils.config_loader import load_agent_config

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUEST = "改正個人情報保護法(APPI 2026)における引受AI(underwriting)の利用目的の特定と同意取得の要件、および少額短期保険(少額短期)の商品開示(disclosure)で必要な注意喚起情報・重要事項について教えてください。"


def _graph_class():
    """Resolve the class exactly as the Pod entry point does -- from cli.py."""
    tree = ast.parse((REPO_ROOT / "cli.py").read_text(encoding="utf-8"))
    name = next(
        n.args[0].id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "run_agent_marketplace"
    )
    module = next(
        n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and any(a.name == name for a in n.names)
    )
    return getattr(importlib.import_module(module), name)


def _invoke(history):
    agent = _graph_class()(config=load_agent_config(REPO_ROOT))
    ctx = InvocationContext(
        caller_id="pb",
        session_id="pb",
        message_id="pb",
        caller_trust_level=TrustLevel.VERIFIED_EXTERNAL,
        request_source="marketplace",
    )
    return agent.invoke(REQUEST, ctx=ctx, input_context={"conversation_history": history})


def _released(state):
    return state.get("formatted_output") or state.get("output") or state.get("result") or ""


class TestTheRunnerSuppliedContextIsAccepted:
    @pytest.mark.parametrize(
        "history",
        [
            pytest.param([], id="first-turn-empty"),
            # get_messages() returns Optional[List[Dict]]: the value can be None.
            pytest.param(None, id="history-unavailable"),
        ],
    )
    def test_the_agent_answers(self, history):
        state = _invoke(history)
        assert str(state.get("status")) == "success", _released(state)
        assert _released(state), "the agent must release something for the caller to read"

    def test_a_real_transcript_does_not_get_screened_as_caller_data(self):
        """Turn two carries this agent's own earlier answer. Accepting the key but
        screening its contents fails here and passes every empty-history test."""
        first = _released(_invoke([]))
        state = _invoke([{"role": "user", "content": REQUEST}, {"role": "assistant", "content": first}])
        assert str(state.get("status")) == "success", _released(state)
        assert _released(state)
