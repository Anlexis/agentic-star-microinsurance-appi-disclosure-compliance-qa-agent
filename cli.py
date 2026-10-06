"""AGENTIC STAR Marketplace entrypoint — one-shot Pod process.

Called by the Dockerfile via CMD ["python", "cli.py"]. Compiles the agent and hands it to
shared.bootstrap.marketplace_app, which owns the Marketplace lifecycle.

The class below must be the one this repo actually defines. The deploy bundle's default
entrypoint hard-codes `Graph`, so a repo whose class is named anything else ImportErrors the
moment the container starts — while build, push and registration all report success.

`namespace` and `agent_name` together resolve the secret path
env/agents/ins/ins_c2_038/, so both are copied from config/agent.yaml rather than
derived from the template id.
"""

from src.graph.graph import MicroinsuranceAppiComplianceQaAgent
from shared.bootstrap.marketplace_app import run_agent_marketplace

if __name__ == "__main__":
    run_agent_marketplace(
        MicroinsuranceAppiComplianceQaAgent,
        agent_name="ins_c2_038",
        namespace="ins",
    )
