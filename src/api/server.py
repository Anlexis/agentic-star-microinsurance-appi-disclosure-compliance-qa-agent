"""AgentCore Platform v1.0"""

# Standalone HTTP entry point for the agent.
#
# Entry points are adapters only — no business logic here. For platform-level
# routing the gateway calls agent.invoke() directly and this module is not used.
#
# Three things this adapter owns, and each exists because of what happens when it
# does not:
#
#   1. Caller authentication. The agent's trust boundary admits only a
#      VERIFIED_EXTERNAL caller, and an unauthenticated caller is ANONYMOUS. With
#      no token configured, every request was therefore refused deep inside the
#      graph and answered 200 with an empty body — a deployment that looks healthy
#      and serves nothing. An unconfigured deployment now refuses at the door with
#      503 and says why.
#
#   2. Credential screening, before invoke(). The framework's output gate scans
#      every value every node returns, and the first node returns the request
#      verbatim — so a credential-shaped string anywhere in the request makes node
#      one fail with a traceback the caller cannot act on. The request cannot
#      succeed either way, so it is refused here with a 400 that names the field.
#      The screen calls the framework's own detector, so the refusal set is
#      exactly the block set enforced one layer later — a local approximation
#      could only drift.
#
#   3. Bounding the session identifier. It travels into correlation and audit
#      records, so it is restricted to a closed alphabet rather than accepted as
#      free text.
#
# 400 is used rather than 422: pydantic owns 422 and returns a list of error
# objects there, so reusing it would make client handling ambiguous.

import os
import secrets
from typing import Any, Dict, cast
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from shared.secrets import factory as secrets_factory
from src.graph.graph import Graph
from src.services.service import detect_output_credentials, is_inert_token

app = FastAPI(title="Agent")

agent = Graph()
agent.compile()
agent.provision_secrets(secrets_factory(namespace="ins", agent_name="MicroinsuranceAppiComplianceQaAgent"))

# Upper bound on the request body's question, mirroring the caller boundary's own
# limit so an oversized body is refused before it is parsed into state.
MAX_INPUT_CHARS = 4000


class InvokeRequest(BaseModel):
    input: str
    session_id: str = ""


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request) -> Dict[str, Any]:
    expected = os.environ.get("INVOKE_AUTH_TOKEN")
    trust = getattr(request.state, "trust_level", TrustLevel.ANONYMOUS)

    if trust is TrustLevel.ANONYMOUS:
        if not expected:
            # Nothing can authenticate a caller, so nothing this endpoint returns
            # could be an answer. Say so once, at the door.
            raise HTTPException(
                status_code=503,
                detail="Caller authentication is not configured on this deployment.",
            )
        supplied = request.headers.get("authorization", "")
        # Compare bytes: compare_digest raises TypeError on non-ASCII str input
        # (headers decode as latin-1), which would 500 instead of the generic 401.
        if not secrets.compare_digest(supplied.encode(), f"Bearer {expected}".encode()):
            # Generic body on purpose — do not leak whether the token was absent,
            # malformed, or wrong.
            raise HTTPException(status_code=401, detail="Token is invalid or expired.")
        trust = TrustLevel.VERIFIED_EXTERNAL

    if len(req.input) > MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Field 'input' exceeds the {MAX_INPUT_CHARS}-character limit.",
        )

    for field, value in (("input", req.input), ("session_id", req.session_id)):
        if detect_output_credentials(value):
            # Name the field, never the value or the pattern's matched text.
            raise HTTPException(
                status_code=400,
                detail=f"Field '{field}' carries a credential-shaped value and was refused.",
            )

    if req.session_id and not is_inert_token(req.session_id):
        raise HTTPException(
            status_code=400,
            detail="Field 'session_id' must be 1-64 characters of [A-Za-z0-9_-].",
        )

    with bound_secrets(agent._secrets_provider):
        ctx = InvocationContext(
            session_id=req.session_id or str(uuid4()),
            caller_trust_level=trust,
            caller_id=getattr(request.state, "caller_id", ""),
        )
        # The framework wheel ships no type information, so invoke() is Any.
        return cast(Dict[str, Any], agent.invoke(req.input, ctx=ctx))


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "agent": "MicroinsuranceAppiComplianceQaAgent"}
