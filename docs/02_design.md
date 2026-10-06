# Template Design Specification — INS-C2-038 Microinsurance APPI 2026 & Product-Disclosure Compliance Q&A Agent

## Position in AgentCore Architecture

- **Agent Class**: MicroinsuranceAppiComplianceQaAgent
- **L1 Base** (framework base class): AgentBaseGraph — direct framework inheritance
- **Pattern**: Cat 2 — RAGAgent (two-layer nested workflow)
- **Three-Layer Separation**:
  - State: flat TypedDict composition (no Pydantic — msgpack incompatible); ADR-005 JSON-serialised strings for all dict/list fields
  - Node: L1 inheritance (Template Method: `execute(self, state: dict) -> dict` override only)
  - Graph: composition (`register_nodes()` for node substitution; Cat 2 nested via `GraphNode`)

## Domain Context

Operator-facing compliance Q&A agent for Japanese 少額短期保険 (small-amount, short-term insurance / "microinsurance") operators. Answers compliance questions across two regulatory layers:

1. **改正個人情報保護法 (APPI 2026)** — conditions for using customer data in underwriting / pricing AI (purpose specification, consent, sensitive-data opt-in, automated-decision transparency, third-party provision for EC-embedded insurance).
2. **金融庁 (FSA) 少額短期保険 product disclosure** — simplified prospectus (契約概要 / 注意喚起情報), mandatory warnings, cooling-off, digital-first disclosure validity, registration & supervisory guidelines.

Answers are **grounded** on a retrieved compliance knowledge base — the agent must not answer beyond the retrieved sources, and cites the passages it used.

## Architecture Overview

### Backbone (outer AgentBaseGraph — fixed 5-node pipeline)

```
START → initialize → pre_process → main(GraphNode) → post_process → finalize → END
                                         ↓ (retry, max 3)
                                       pre_process
```

### Inner Domain Workflow (DomainWorkflowGraph — linear 5-node RAG pipeline)

```
START → input_validate → retrieve → rerank_filter → generate_answer → output_format → END
```

### Node Configuration

| Node | Class | File | Trust | Responsibility | Input Keys | Output Keys |
|------|-------|------|-------|---------------|------------|-------------|
| initialize | InitializeNode | framework | — | session init | — | session_id, schema_version |
| pre_process | PreProcessNode | src/nodes/pre_process_node.py | VERIFIED_EXTERNAL | trust boundary + caller contract: instruction-override screen, personal-data mask, length bound | user_input | validated_input, enriched_context |
| main | ComplianceQaGraphNode | src/graph/graph.py | — | delegates to DomainWorkflowGraph | validated_input | qa_answer, citations, retrieved_count, result |
| post_process | PostProcessNode | src/nodes/post_process_node.py | ANONYMOUS | output boundary: credential scan, disclaimer invariant, clears every answer-bearing field on a withholding | result | formatted_output, result, qa_answer, citations |
| finalize | FinalizeNode | framework | — | response metadata | — | response_metadata, total_time_ms |
| input_validate (inner) | InputValidateNode | src/nodes/input_validate_node.py | ANONYMOUS | normalise question + topic classification | user_input (inner channel) | qa_query |
| retrieve (inner) | RetrieveNode | src/nodes/retrieve_node.py | ANONYMOUS | retrieve top_k knowledge-base passages | qa_query, runtime_settings | retrieved_passages, retrieved_count |
| rerank_filter (inner) | RerankFilterNode | src/nodes/rerank_filter_node.py | ANONYMOUS | rerank + apply the relevance floor | retrieved_passages, runtime_settings | reranked_passages, retrieved_count |
| generate_answer (inner) | GenerateAnswerNode | src/nodes/generate_answer_node.py | ANONYMOUS | grounded answer assembled from passages (deterministic — no model invoked) | reranked_passages, qa_query, runtime_settings | qa_answer, citations |
| output_format (inner) | OutputFormatNode | src/nodes/output_format_node.py | ANONYMOUS | assemble answer + sources block | qa_answer, citations | result, qa_answer |

### Data Flow

```
user_input (compliance question, plain text)
    │
    ▼ PreProcessNode (VERIFIED_EXTERNAL, S-1/S-2)
validated_input (normalised question)
enriched_context (JSON string — ADR-005)
    │
    ▼ ComplianceQaGraphNode → DomainWorkflowGraph
    │   InputValidateNode  → qa_query (JSON: question, normalized, topics)
    │   RetrieveNode        → retrieved_passages (JSON), retrieved_count
    │   RerankFilterNode    → reranked_passages (JSON), retrieved_count
    │   GenerateAnswerNode  → qa_answer (str), citations (JSON)   [grounded]
    │   OutputFormatNode    → result (str), qa_answer (str)
    ▼ merge_output
qa_answer, citations, retrieved_count, result → outer state
    │
    ▼ PostProcessNode (ANONYMOUS, S-3)
formatted_output (S-3-gated result + mandatory disclaimer)
```

### State Definition

| Field | Type | Purpose | Producer |
|-------|------|---------|----------|
| validated_input | NotRequired[Optional[str]] | Screened, masked, normalised question | PreProcessNode |
| runtime_settings | NotRequired[Optional[str]] | JSON: bounds-checked declared settings {top_k, score_threshold, hybrid_search, system_prompt_template} | DomainWorkflowGraph._extra_initial_state |
| enriched_context | NotRequired[Optional[str]] | JSON: {source, channel} | PreProcessNode |
| qa_query | NotRequired[Optional[str]] | JSON: {question, normalized, topics} | InputValidateNode |
| retrieved_passages | NotRequired[Optional[str]] | JSON: candidate passages [{id,title,text,source,hits}] | RetrieveNode |
| reranked_passages | NotRequired[Optional[str]] | JSON: filtered passages [{id,title,text,source,score}] | RerankFilterNode |
| retrieved_count | NotRequired[Optional[int]] | Count of grounded passages kept | Retrieve / RerankFilter |
| qa_answer | NotRequired[Optional[str]] | Grounded answer text | GenerateAnswerNode |
| citations | NotRequired[Optional[str]] | JSON: [{id,title,source}] | GenerateAnswerNode |
| result | NotRequired[Optional[str]] | Answer + sources block | OutputFormatNode |

**ADR-005 constraint**: all dict/list-valued fields use JSON-serialised `Optional[str]`. `to_json()` / `from_json()` helpers are defined in `src/schemas/state.py` and used at every producer/consumer boundary — one contract end-to-end.

**Prohibited**: re-declaring `formatted_output` (inherited from AgentState), credentials in State, Pydantic models.

### Input / Output Contract

- **Input** (`user_input`): a natural-language compliance question (plain text). Matches `deploy/invoke_payload.json`.
- **Output** (`formatted_output`): grounded answer + citations block + mandatory compliance disclaimer; or a grounded-refusal message when no source passage clears the retrieval threshold.

## Security Configuration

| Layer | Gate | Implementation |
|-------|------|---------------|
| S-1 | Trust enforcement | PreProcessNode `required_trust_level = VERIFIED_EXTERNAL`; inner nodes ANONYMOUS |
| S-2 | Input validation | PreProcessNode owns the caller contract: instruction-override and chat-template control-token screening (raw and markup-stripped), personal-data masking with explicit character guards, and a length bound. Enforced in the node, provable by calling `execute()` directly. |
| S-3 | Output gate | PostProcessNode is the output boundary. Credential detection delegates to the framework's own detector, so the refusal set is exactly the block set enforced one layer later; the mandatory disclaimer is enforced as an invariant of the released answer; on a violation every answer-bearing field is blanked and a truthy withheld notice naming a closed-set reason is published. The agent class carries the same check as a second layer. |
| S-4 | Audit logging | `emit_trace_event()` (positional) in every node's `execute()` |
| S-5 | Credential handling | No credentials in State; secrets via InvocationContext only |

### Runtime configuration

`config/agent.yaml` is the flat manifest — identity, entry point, trust level and the
compile-time `requires` block. It carries no runtime settings. Everything the running agent reads
lives in **`config/config.yaml`**:

```yaml
llm:
  system_prompt_template: "prompts/appi_compliance_qa.j2"
retrieval:
  top_k: 8
  score_threshold: 0.75
  hybrid_search: true
security:
  s3_gate_enabled: true
```

Rationale: compliance answers decide disclosure obligations, so the relevance floor is high
(`score_threshold: 0.75`) and a weakly-matching passage is not treated as grounding; the corpus is
long regulatory text, so `top_k: 8`; article numbers and product codes appear in passage text but
not in every keyword list, so `hybrid_search: true`.

`temperature` and `max_tokens` are deliberately **absent**. The manifest declares
`generation_mode: deterministic` — the answer body is assembled from retrieved passage text and no
model is invoked — so no reader for them can exist, and a declared value nothing reads is worse
than no declaration.

**How a declared value reaches a node.** The framework calls `node.execute(state)` with one
argument, so a node cannot be handed the graph's config directly. `ComplianceQaGraphNode._parent_config()`
reads `config/config.yaml`, bounds-checks each value, and forwards the survivors; the inner graph
serialises them into the single `runtime_settings` state channel, which is the only route a node
reads them from. A value outside its range is dropped rather than clamped, and the consuming node
falls back to its own documented default.

## Framework Utilization

### Shared Components Used
- [x] InvocationContext (`config["configurable"]` — session_id, trust_level, retrieval knobs)
- [x] S-3: agent-class `_security_gate_output()` + module-level scan in `post_process_node.py`
- [x] S-4: `emit_trace_event()` — at least one domain-specific event per node `execute()` (positional form)
- [x] `to_json()` / `from_json()` helpers in `src/schemas/state.py` — ADR-005 serialisation contract

### Composition Pattern

- **Pattern**: Cat 2 nested two-layer — GraphNode wrapping inner BaseGraph
- **Outer graph**: `MicroinsuranceAppiComplianceQaAgent(AgentBaseGraph)` — fixed 5-node backbone
- **Inner graph**: `DomainWorkflowGraph(BaseGraph)` — 5-node linear RAG pipeline
- **Error propagation**: propagate (SubgraphError on inner failure; outer backbone retries pre_process)

## Import Isolation Confirmation
- [x] Template does not import the Level-0 platform SDK
- [x] Import targets: `framework/` and `shared/` only (no Level-0 SDK)

## Design Decision Record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| L1 base type | AgentBaseGraph | AutonomousBaseGraph | AgentBaseGraph | Fixed sequential RAG pipeline; no reasoning loop required |
| Composition pattern | Cat 1 (flat) | Cat 2 (nested GraphNode) | Cat 2 nested | 5 sequential RAG steps; RAGAgent pattern |
| Retrieval (v1) | Real vector store | Rule-based hybrid stub | Rule-based hybrid stub (v1) | No managed retrieval store in SDK v1.0.0rc1; score contract preserved for production swap |
| Grounding | Free generation | Passage-grounded only | Passage-grounded only | Compliance domain — must not hallucinate; refuse when no source clears threshold |
| State dict fields | bare dict | JSON-serialised str | JSON-serialised str | ADR-005: msgpack serialisation safety |
| S-3 gate location | node instance hook | agent class + module fn | agent class + module fn | An `_extra_*` instance hook is auto-wrapped by the real SDK and fails at runtime |
| Envelope containment | override `get_output()` | clear at the boundary | clear at the boundary | The base envelope resolves `formatted_output or result`, so an override blanking `output` on a non-success status would contain the leak on its own — and make the boundary's own clearing unfalsifiable. One layer, where the leak surfaces. |
| Question echoed into the answer | render verbatim | neutralise then render | neutralise then render | The answer is a cited rendering; a newline in the question could otherwise open a line that reads as a retrieved source and attribute invented text to a real regulator |
