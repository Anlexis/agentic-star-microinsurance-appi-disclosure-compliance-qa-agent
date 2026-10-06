# Test Specification — INS-C2-038 Microinsurance APPI 2026 & Disclosure Compliance Q&A Agent

## 1. Test Strategy

- **Agent:** INS-C2-038 — Microinsurance APPI 2026 & Product-Disclosure Compliance
  Q&A (Cat 2, **RAG pattern**, two-layer nested graph: outer `AgentBaseGraph`
  backbone + inner `DomainWorkflowGraph` `BaseGraph`).
- **Coverage target:** ≥ 90% of `src/nodes/` + `src/graph/` branches.
- **Test types:** Unit (per node + graph wiring) · Proof-of-Boundary (framework
  security/serialization contracts) · Backbone invoke (full `Graph().invoke()`).
- **Framework provisioning:** `framework` (agenticstar-agentcore) is supplied by
  CI — the CI stub package on `PYTHONPATH` for the CI stub package arm, or the wheel from the
  package registry for the wheel arm. Tests import the REAL Wave-1 modules on
  `develop`; there are no stub nodes.
- **S-4 audit:** `emit_trace_event` is patched at the node module level in unit
  tests to avoid audit-backend calls, never via a `sys.modules` stub (which would
  break the real `shared` package the framework loads at import time).

### Test file map

| File | Scope |
|------|-------|
| `tests/unit/test_nodes.py` | All 7 domain/backbone nodes + outer & inner graph wiring |
| `tests/unit/test_main_node.py` | Deprecated `MainNode` stub — `execute()` contract kept green |
| `tests/proof_of_boundary/test_pb_invoke_order.py` | PB-6 per-node + backbone invoke order (VERIFIED_EXTERNAL) + S-1 gate + payload alignment |
| `tests/proof_of_boundary/test_import_isolation.py` | PB-4 Level-0 import isolation (AST scan) |
| `tests/proof_of_boundary/test_state_safety.py` | PB-2/PB-5 State msgpack/credential safety (AST scan) |
| `tests/proof_of_boundary/test_pb7_hitl_interrupt_propagation.py` | PB-7 HITL interrupt-propagation (skip stub — no cross-boundary HITL) |

### Canonical valid payload (PB-6 `_VALID_PAYLOAD`)

INS-C2-038 takes a **natural-language compliance question** (plain text — NOT a
JSON object). The canonical question used by the backbone invoke test and by
`deploy/invoke_payload.json` (the two MUST stay identical — asserted by
`test_invoke_payload_matches_pb6`):

```
改正個人情報保護法(APPI 2026)における引受AI(underwriting)の利用目的の特定と
同意取得の要件、および少額短期保険(少額短期)の商品開示(disclosure)で必要な
注意喚起情報・重要事項について教えてください。
```

Grounding: the question carries strong APPI-2026 (`引受` / `underwriting` /
`利用目的` / `同意`) and SSTI-disclosure (`少額短期` / `開示` / `注意喚起`)
keyword coverage, so `RetrieveNode` returns ≥ 1 KB passage ⇒ `GenerateAnswerNode`
produces a **grounded** answer (never the abstention path) ⇒ SUCCESS end-to-end.

## 2. RAG Quality Tests — Grounding & Abstention (Mandatory)

The two non-negotiable RAG contracts for this template are **grounding** (answers
are synthesised strictly from retrieved passages, always cited) and **abstention**
(when nothing relevant is retrieved, refuse rather than hallucinate).

| TC-ID | Test | Expected Result | Where |
|-------|------|----------------|-------|
| RG-01 | Grounded answer cites only the supplied passages | `citations` == the reranked passage ids; each passage body appears in `qa_answer` | `TestGenerateAnswerNode::test_grounded_answer_cites_only_supplied_passages` |
| RG-02 | Sources block rendered from citations | `参考資料 / Sources:` + `[<id>]` markers present in `result` | `TestOutputFormatNode::test_assembles_result_with_sources_block` |
| RG-03 | End-to-end grounded run | full invoke `output` contains the grounded answer, the sources block, and the mandatory disclaimer | `TestBackboneInvokeOrder::test_backbone_invoke_succeeds_and_returns_output` |
| AB-01 | Abstention on no grounded passages | `qa_answer` == `UNGROUNDED_ANSWER`; `citations` == `[]`; `status=success` | `TestGenerateAnswerNode::test_ungrounded_abstains_rather_than_inventing` |
| AB-02 | Abstention end-to-end | inner graph invoke of an off-topic question → `retrieved_count=0` + refusal answer | `TestInnerDomainGraph::test_inner_graph_abstains_when_ungrounded` |
| RT-01 | Retrieval keyword grounding | APPI-keyword query returns `APPI-01`; passages sorted by descending hits | `TestRetrieveNode` |
| RT-02 | Declared `top_k` caps what is carried forward | `retrieved_count == top_k` for a query matching more passages | `TestRetrieveNode::test_declared_top_k_caps_results` |
| RT-03 | An out-of-range `top_k` cannot take effect | 0 / negative / NaN / Infinity / bool / non-numeric all fall back to the node default | `TestRetrieveNode::test_out_of_range_top_k_falls_back_to_the_node_default` |
| RT-04 | Declared `hybrid_search` changes the retrieved set | scoring on passage text as well as the keyword list returns different hit counts | `TestRetrieveNode::test_hybrid_search_changes_the_retrieved_set` |
| RK-01 | Rerank hits → [0,1] score + threshold filter | scores normalised to the top passage; passages below `score_threshold` dropped | `TestRerankFilterNode` |
| RK-02 | A below-floor passage is never promoted | nothing clearing the floor yields nothing; there is no top-1 fallback | `TestRerankFilterNode::test_a_below_floor_passage_is_never_promoted` |
| RK-03 | A non-finite or out-of-range floor cannot take effect | NaN / ±Infinity / bool / out-of-range all fall back to the node default | `TestRerankFilterNode::test_non_finite_or_out_of_range_threshold_falls_back` |
| RK-04 | Declared settings are live end to end | changing `score_threshold` or `hybrid_search` in `config/config.yaml` changes what `/invoke` returns | `TestOuterGraphComposition::test_declared_settings_are_forwarded_when_valid`, `TestInnerDomainGraph::test_seeds_the_declared_settings_into_state` |
| TP-01 | Topic tagging (APPI / SSTI / general) | `appi_2026` / `ssti_disclosure` / `general_compliance` per keywords | `TestInputValidateNode` |

## 3. Framework Compliance Tests (Mandatory)

| TC-ID | Test | Expected Result | Where |
|-------|------|----------------|-------|
| TC-01 | State contract: flat `TypedDict`, domain fields `NotRequired`, no Pydantic/dataclass | AST scan: 0 violations | `test_state_safety.py` |
| TC-02 | Empty / non-string / overlong input rejected at PreProcessNode | `status=error`, error_log populated | `TestPreProcessNode` |
| TC-02a | Instruction-override and chat-template control tokens refused BY THE NODE | `execute()` called directly returns `status=error` and no `validated_input` | `TestPreProcessOwnsTheRefusal` |
| TC-02b | The screens do not fire on real compliance text | every corpus sentence and the canonical question pass | `TestScreensDoNotFireOnRealComplianceText` |
| TC-02c | Personal data masked regardless of script | an individual number written against a Kanji is masked exactly as the space-separated form | `TestPersonalDataMasking` |
| TC-02d | A refusal names the field, never the value | the rejected string appears nowhere in the delta | `test_a_refusal_never_echoes_what_was_rejected` |
| TC-03 | No JWT/credential in State | CI `gate-credential-scan`: 0 violations | CI + `test_state_safety.py` |
| TC-04 | `execute(self, state)` contract — no `_invoke_impl` | Signature `(self, state)`, `_invoke_impl` absent | `test_execute_signature_is_state_first`, `test_main_node.py` |
| TC-05 | S-4: `emit_trace_event()` called inside each node `execute()` | ≥1 domain event per node (positional form) | `scripts/check_audit_trace.py` |
| TC-08 | S-1: `required_trust_level` enforced in `__call__` before `execute()` | ANONYMOUS caller → refused; VERIFIED_EXTERNAL → admitted | `TestS1TrustGate` |
| TC-08a | Outer `PreProcessNode` = VERIFIED_EXTERNAL; inner nodes + post_process = ANONYMOUS | trust levels asserted per node | `test_trust_level_*` |
| TC-11 | Output boundary on post_process | every credential shape the framework detects is withheld; clean answers are released with the disclaimer | `TestPostProcessNode` |
| TC-11a | The local refusal set is a superset of the framework's | anything `detect_credentials` finds, the boundary finds | `TestCredentialDetectionMatchesTheFramework::test_the_local_set_is_a_superset_never_a_subset` |
| TC-11b | A withholding clears every answer-bearing field | each field PRESENT in the delta and empty — an omitted key leaves the old value in state | `test_every_credential_shape_the_framework_knows_is_withheld` |
| TC-11c | The withheld notice is truthy | a falsy replacement re-opens the envelope's fallback | `test_the_withheld_notice_is_truthy` |
| TC-11d | The mandatory disclaimer is an invariant of the released answer | an answer reaching the boundary without it is withheld | `test_an_answer_without_the_disclaimer_is_withheld` |

## 4. Proof-of-Boundary Tests (Mandatory)

| PB-ID | Boundary | Test | Expected Result | Where |
|-------|----------|------|----------------|-------|
| PB-2 | State serialization | AST scan of `src/schemas/state.py` | primitives only; no Pydantic/dataclass | `test_state_safety.py` |
| PB-4 | Import isolation | AST scan of `src/` | 0 Level-0 (`agenticstar` / platform) imports | `test_import_isolation.py` |
| PB-5 | Checkpoint safety | no credential-named fields / prohibited types in State | inspection pass | `test_state_safety.py` |
| PB-6 | Invoke execution order (per node) | `__call__`: S-4 node_start → S-1 gate → S-2 input gate → `execute()` → S-3 output gate → S-4 node_complete | order verified for every `src/nodes/` class | `TestInvokeOrder` |
| PB-6b | Backbone invoke order | full `Graph().invoke(_VALID_PAYLOAD, ctx=VERIFIED_EXTERNAL)` | `status=success`; node_history = `[Initialize, PreProcess, ComplianceQaGraphNode, PostProcess, Finalize]` | `TestBackboneInvokeOrder` |
| PB-6c | Real external caller | `InvocationContext(caller_trust_level=VERIFIED_EXTERNAL)` — **never** `for_internal()` | inner ANONYMOUS nodes accept the passthrough trust; SUCCESS end-to-end | `TestBackboneInvokeOrder` |
| PB-6d | Payload alignment | `deploy/invoke_payload.json["input"] == _VALID_PAYLOAD` | Stage-5 deploy-stg invoke exercises the PB-6 payload | `test_invoke_payload_matches_pb6` |
| PB-7 | HITL interrupt propagation | skip stub — `propagate_hitl=False`, no cross-boundary interrupt() checkpoint | skipped with reason (real assertion when HITL wired) | `test_pb7_hitl_interrupt_propagation.py` |
| PB-8 | Output boundary, end to end through the real HTTP entry point | a clean request still returns its real answer and the boundary node appears in `node_history` | `TestCleanPathControl` |
| PB-8a | An answer channel the boundary cannot read is withheld, not raised | with a drifted `merge_output` on the data path the caller receives a truthy notice, not the pre-gate payload | `TestAnswerChannelTheBoundaryCannotRead` |
| PB-8b | Error paths carry no answer and no internal detail | refused requests return `output: null`, no traceback, no source paths | `TestErrorPathsCarryNoAnswer` |
| PB-8c | An unconfigured deployment refuses at the door | no `INVOKE_AUTH_TOKEN` → 503 | `test_an_unauthenticated_caller_is_refused_at_the_door` |
| PB-9 | Caller data cannot forge a citation | a question carrying newlines and citation markup renders no source line and no sources entry | `TestCallerDataCannotForgeACitation` |
| PB-10 | Credential-shaped input is refused readably | 400 naming the field, never the value | `TestCredentialShapedInputIsRefusedReadably` |

## 5. Graph Composition Tests (Cat 2 nested)

| GC-ID | Test | Expected Result | Where |
|-------|------|----------------|-------|
| GC-01 | Outer backbone registers 5 slots | `{initialize, pre_process, main, post_process, finalize}`; main = `ComplianceQaGraphNode` | `TestOuterGraphComposition::test_registers_five_backbone_slots` |
| GC-02 | Agent identity | `name == "MicroinsuranceAppiComplianceQaAgent"`; `state_schema is State`; `Graph is` the real class | `TestOuterGraphComposition` |
| GC-03 | Main-slot GraphNode contracts | `error_strategy="propagate"`; `propagate_hitl=False`; `extract_input` takes ONLY the screened question — never the raw `user_input` | `test_main_slot_contracts`, `test_extract_input_takes_only_the_screened_question` |
| GC-04 | Graph key coupling | inner `get_output` ↔ outer `merge_output` — 5 coupled keys (`qa_answer`, `citations`, `retrieved_count`, `result`, `status`); `merge_output` returns changed keys only | `test_merge_output_maps_the_coupled_keys_only` |
| GC-06 | No `get_output` in this repository resolves its answer field with `or` | the base envelope already does; a second one a level down would be the same defect in code that is ours to delete | `test_no_get_output_override_reopens_a_fallback` |
| GC-05 | Inner graph registers 5 RAG nodes | `{input_validate, retrieve, rerank_filter, generate_answer, output_format}` | `TestInnerDomainGraph::test_registers_five_domain_nodes` |

### Negative / boundary cases

| Case | Node | Expected |
|------|------|----------|
| empty `user_input` | PreProcessNode | `status=error`, "empty" |
| non-string `user_input` | PreProcessNode | `status=error` |
| input > 4000 chars | PreProcessNode | `status=error`, length reason |
| instruction-override / control token | PreProcessNode | `status=error`, screened reason |
| credential-shaped `input` or `session_id` | HTTP entry point | 400 naming the field |
| `session_id` outside `[A-Za-z0-9_-]{1,64}` | HTTP entry point | 400 |
| no question on the inner channel | InputValidateNode | `status=error`, no `qa_query` |
| empty normalised query | RetrieveNode | `retrieved_count=0`, `status=success` |
| empty retrieved passages | RerankFilterNode | `retrieved_count=0`, `status=success` |
| no grounded passages | GenerateAnswerNode | abstention text, `citations=[]`, `status=success` |
| missing `qa_answer` | OutputFormatNode | `status=error`, "qa_answer" |
| empty `result` | PostProcessNode | no-grounding reply + disclaimer, `status=success` |
| credential shape in the answer | PostProcessNode | withheld, answer fields blanked, `status=error` |
| non-string `result` | PostProcessNode | withheld with a reason code — never raised |
| answer missing the disclaimer | PostProcessNode | withheld, `status=error` |

## 6. Test Execution Summary

- Execution: `pytest tests/` — 264 passed, 1 skipped (the PB-7 stub).
- RAG contracts covered: **grounding** (RG-01–03, RT/RK) and **abstention**
  (AB-01/02) — the two mandatory RAG behaviours for this template.
- Gates: `gate-dep-pinning`, `gate-stub-check`, `gate-cat-consistency`,
  import-isolation, composition, invoke-chain, credential-scan, trust-level,
  scaffold-integrity — all PASS.
- Coverage: node + graph modules exercised on both success and error paths.
