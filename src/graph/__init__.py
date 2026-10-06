"""AgentCore Platform v1.0"""

# AgentRegistry entry point (source-review fix, SR High #12).
# config/agent.yaml declares:
#     module: "src.graph"
#     class:  "MicroinsuranceAppiComplianceQaAgent"
# so AgentRegistry resolves the class via
#     getattr(import_module("src.graph"), "MicroinsuranceAppiComplianceQaAgent").
# This package previously exported only a docstring, so that lookup failed and
# manifest-based lazy-load was broken. Re-export the agent class (and the
# `Graph` alias server.py imports) from the graph submodule.
from src.graph.graph import Graph, MicroinsuranceAppiComplianceQaAgent

__all__ = ["MicroinsuranceAppiComplianceQaAgent", "Graph"]
