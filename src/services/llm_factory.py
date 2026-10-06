"""AgentCore Platform v1.0"""

# Builds the LLM client the platform actually provisions: Azure OpenAI.
#
# Why this file exists: the Marketplace entry point constructs the agent with no
# config, so a node that takes its client from `config["llm"]` receives None on
# every invocation and silently falls through to its deterministic branch. An
# agent designed around an LLM then never uses one, with no error and nothing in
# the response to show for it. Building the client here — from the credentials
# the platform injects — is what actually puts an LLM behind the agent.
#
# Why Azure only: the registered environment supplies AZURE_OPENAI_API_KEY,
# AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT, and the image ships the
# `openai` extra. No other provider's credentials or client library are present,
# so probing for them would be dead code. One template keeps a provider-probing
# variant for the case where a different credential is issued.
#
# Returning None is a supported outcome, not a failure: the deterministic branch
# is a working path, and taking an agent down over a packaging or configuration
# problem it cannot fix would be worse than degrading.

from typing import Any

from shared.services.llm.base_llm import BaseLLM

_REQUIRED = ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT")


class _FleetShim:
    """Adapts BaseLLM to the call shape these nodes already use.

    The substrate contract is `complete(messages: list) -> dict` returning
    `{"content", "tool_calls", "model", "usage"}`. The call sites here predate
    that: they pass a bare prompt string plus per-call kwargs and use the result
    as a string.

    Bridging in one place rather than rewriting call sites is deliberate — the
    call sites are domain logic and would each need re-verifying, while this seam
    is a single file that stays correct if the provider changes.

    `api_key` is accepted and ignored: the key is already bound into the client
    at construction. Keeping the parameter means existing callers do not change.
    """

    def __init__(self, inner: BaseLLM) -> None:
        self._inner = inner

    def complete(self, prompt: str, api_key: str | None = None, **_: Any) -> str:
        # Per-call temperature/max_tokens are dropped: BaseLLM binds those at
        # construction. Losing them changes cost, not correctness. A node that
        # genuinely needs per-call sampling should build its own client.
        result = self._inner.complete([{"role": "user", "content": prompt}])
        return str(result.get("content", ""))


def build_llm(secrets: Any) -> _FleetShim | None:
    """Return a client built from the invocation's secrets, or None.

    None means "no LLM configured for this invocation" and callers must keep
    handling it — every call site retains its deterministic branch.
    """
    values = [secrets.get(name) for name in _REQUIRED]
    if not all(values):
        return None
    key, endpoint, deployment = values

    try:
        from shared.services.llm.azure_openai_client import AzureOpenAIClient
    except ImportError:
        # `langchain-openai` absent from the image. Treated as unconfigured
        # rather than raised: a credential we cannot build a client for is the
        # same as no credential, and the deterministic path still works.
        return None

    # azure_endpoint must be the BARE resource endpoint — the client raises on
    # anything containing '/openai'. Pass the configured value straight through
    # and let the client reject it loudly rather than rewriting it here; a
    # published env table has been seen carrying a full API path.
    return _FleetShim(
        AzureOpenAIClient(
            {
                "api_key": key,
                "azure_endpoint": endpoint,
                "azure_deployment": deployment,
                "temperature": 0.0,
                "max_tokens": 1024,
            }
        )
    )


def configured_provider(secrets: Any) -> str | None:
    """Which provider build_llm() would use — for audit events and tests."""
    return "azure" if all(secrets.get(name) for name in _REQUIRED) else None


def resolve_llm(injected: Any, state: Any) -> tuple[Any, str | None]:
    """The client to use for this invocation, plus the provider label for audit.

    Prefers a client the caller injected (the standalone path supplies one);
    otherwise builds one from this invocation's secrets. Returns (None, None) when
    no LLM is configured — a supported mode, so callers keep their deterministic
    branch.

    The result is deliberately returned rather than stored: node instances are
    shared through the registry LRU cache, so a client built from one caller's
    secrets must not outlive that call.
    """
    if injected is not None:
        return injected, "injected"
    try:
        from framework.schemas.invocation_context import InvocationContext

        client = build_llm(InvocationContext.from_state(state).secrets)
    except Exception:  # noqa: BLE001 — no client is a supported mode, not a fault
        return None, None
    # The label is derived from the client, not from credential presence: an image
    # without the client library yields credentials that resolve and a client that
    # does not, and reporting "azure" there would describe an LLM call that never
    # happened.
    return client, ("azure" if client is not None else None)
