"""AgentCore Platform v1.0"""

# INS-C2-038 — RetrieveNode
#
# Inner domain node 2: retrieve candidate passages from the compliance knowledge
# base (APPI 2026 and the 少額短期保険 product-disclosure supervisory guidelines).
#
# The shipped build scores passages lexically over the knowledge base bundled in
# this module. A production build replaces _retrieve() with a search against a
# managed retrieval store; the per-passage `hits` contract is preserved, so no
# downstream node changes.
#
# Two declared settings are read here, and both are read from state:
#   top_k          — how many passages are carried forward
#   hybrid_search  — when true, a passage also scores on its title and body text,
#                    not only on its keyword list. The knowledge base carries law
#                    article numbers and product codes that appear in the text and
#                    not in the keywords, so the two modes return different sets.
#
# Inner node: ANONYMOUS trust (the trust boundary is PreProcessNode).
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import re
from typing import Any, ClassVar, Dict, List, Set

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import from_json, to_json
from src.services.service import finite_in_range

logger = logging.getLogger(__name__)

# Used when top_k is absent from the declared settings or failed its bounds check.
DEFAULT_TOP_K = 8
_TOP_K_BOUNDS = (1, 64)

# Used when hybrid_search is absent or is not a boolean.
DEFAULT_HYBRID_SEARCH = True

# Term extraction for the hybrid signal. ASCII runs are words; Japanese runs are
# cut into overlapping trigrams, because the script carries no word separators.
_ASCII_TERM_RE = re.compile(r"[a-z0-9]{4,}")
_CJK_RUN_RE = re.compile(r"[一-鿿぀-ゟ゠-ヿ]{3,}")

# Compliance knowledge base (read-only; never mutated in execute()).
# Each passage: {id, title, text, source, keywords}
_KNOWLEDGE_BASE: List[Dict[str, Any]] = [
    {
        "id": "APPI-01",
        "title": "APPI 2026 — 引受AIにおける利用目的の特定と同意",
        "text": (
            "改正個人情報保護法(APPI 2026)は、引受(underwriting)・料率算定AIに"
            "顧客データを用いる場合、利用目的を可能な限り具体的に特定し、必要な範囲で"
            "本人の同意を取得することを求めます。目的外利用は禁止され、目的変更時は"
            "再同意または通知・公表が必要です。"
        ),
        "source": "APPI 2026 / 個人情報保護委員会ガイドライン",
        "keywords": [
            "appi",
            "個人情報",
            "利用目的",
            "purpose",
            "underwriting",
            "引受",
            "consent",
            "同意",
            "pricing",
            "料率",
        ],
    },
    {
        "id": "APPI-02",
        "title": "APPI 2026 — 要配慮個人情報とリスクスコアリング",
        "text": (
            "健康情報等の要配慮個人情報(sensitive data)をリスクスコアリングに用いる"
            "場合、原則としてオプトインの明示同意が必要です。第三者提供のオプトアウトは"
            "認められず、取得・利用の記録保持(監査可能性)が求められます。"
        ),
        "source": "APPI 2026 / 要配慮個人情報の取扱い",
        "keywords": [
            "appi",
            "要配慮",
            "sensitive",
            "health",
            "健康",
            "risk",
            "リスク",
            "スコアリング",
            "scoring",
            "opt-in",
            "同意",
        ],
    },
    {
        "id": "APPI-03",
        "title": "APPI 2026 — 自動化された意思決定の透明性",
        "text": (
            "引受判断が自動化された意思決定(automated decision / profiling)による"
            "場合、本人に対する説明可能性(transparency)と、判断ロジックの概要開示、"
            "および人間による再審査の窓口整備が推奨されます。"
        ),
        "source": "APPI 2026 / 自動化意思決定に関する考え方",
        "keywords": [
            "appi",
            "automated",
            "自動化",
            "profiling",
            "プロファイリング",
            "transparency",
            "説明",
            "意思決定",
            "decision",
        ],
    },
    {
        "id": "APPI-04",
        "title": "APPI 2026 — 第三者提供とEC組込保険のデータ共有",
        "text": (
            "EC組込型(embedded)保険で、プラットフォーム事業者(大手EC・通信事業者"
            "等)と顧客データを共有する場合、第三者提供の規律が"
            "適用されます。共同利用の要件明示、または本人同意の取得と提供記録の作成・"
            "保存が必要です。"
        ),
        "source": "APPI 2026 / 第三者提供・共同利用",
        "keywords": [
            "appi",
            "third party",
            "第三者提供",
            "embedded",
            "組込",
            "ec",
            "rakuten",
            "amazon",
            "共同利用",
            "data sharing",
        ],
    },
    {
        "id": "SSTI-01",
        "title": "少額短期保険 — 簡易目論見書(契約概要・注意喚起情報)",
        "text": (
            "少額短期保険の商品開示(product disclosure)では、契約概要と注意喚起情報を"
            "簡易かつ平易な表現で交付する必要があります。重要事項説明では、保障内容・"
            "保険期間・保険金額の上限・免責事由を明確に記載します。"
        ),
        "source": "FSA 少額短期保険 監督指針 / 商品開示",
        "keywords": [
            "少額短期",
            "ssti",
            "disclosure",
            "開示",
            "prospectus",
            "契約概要",
            "注意喚起",
            "重要事項",
            "product",
            "簡易",
        ],
    },
    {
        "id": "SSTI-02",
        "title": "少額短期保険 — 必要な警告表示とクーリングオフ",
        "text": (
            "少額短期保険では、保険金額の上限・保険期間の短さ・主要な免責事由を警告"
            "(warning)として目立つ形で表示します。クーリングオフ(cooling-off)の"
            "適用可否と手続を、申込時点で明確に案内する必要があります。"
        ),
        "source": "FSA 少額短期保険 監督指針 / 注意喚起",
        "keywords": [
            "少額短期",
            "warning",
            "警告",
            "クーリングオフ",
            "cooling",
            "免責",
            "exclusion",
            "上限",
            "cap",
            "保険期間",
        ],
    },
    {
        "id": "SSTI-03",
        "title": "少額短期保険 — デジタル完結開示(電子交付)の有効性",
        "text": (
            "デジタルファースト(digital-first)の電子交付(electronic delivery)は、"
            "本人が電磁的方法による受領に同意し、内容を確実に閲覧・保存できる状態を"
            "確保した場合に有効(validity)とされます。EC組込チャネルでも、開示到達の"
            "証跡保存が求められます。"
        ),
        "source": "FSA 少額短期保険 監督指針 / 電子交付",
        "keywords": [
            "少額短期",
            "digital",
            "電子交付",
            "電子",
            "electronic",
            "delivery",
            "有効",
            "validity",
            "ec",
            "embedded",
            "同意",
        ],
    },
    {
        "id": "SSTI-04",
        "title": "少額短期保険 — 登録・監督とAPPIの二層コンプライアンス",
        "text": (
            "少額短期保険業者は保険業法に基づく登録(registration)を要し、FSA"
            "(金融庁)の監督指針に従います。データ利活用はAPPIの規律と重畳的に適用"
            "されるため、商品開示規制とAPPI 2026の双方を満たす二層のコンプライアンス"
            "体制が必要です。"
        ),
        "source": "FSA / 保険業法・少額短期保険 監督指針",
        "keywords": [
            "少額短期",
            "fsa",
            "金融庁",
            "監督指針",
            "registration",
            "登録",
            "保険業法",
            "supervisory",
            "compliance",
            "appi",
        ],
    },
]


def _question_terms(low: str) -> Set[str]:
    """Break a question into comparable terms without relying on spaces.

    Japanese is written without word separators, so splitting on whitespace yields
    one enormous token that matches nothing. ASCII runs are taken as words;
    Japanese runs contribute overlapping character trigrams, which is the
    equivalent unit — short enough to match a term used in a different
    construction, long enough not to match on a single common character.
    """
    terms: Set[str] = set(_ASCII_TERM_RE.findall(low))
    for run in _CJK_RUN_RE.findall(low):
        terms.update(run[i : i + 3] for i in range(len(run) - 2))
    return terms


def _retrieve(
    normalized_question: str,
    top_k: int,
    hybrid_search: bool = DEFAULT_HYBRID_SEARCH,
) -> List[Dict[str, Any]]:
    """Score every knowledge-base passage against the question; return the best.

    A passage always scores on its curated keyword list. When *hybrid_search* is
    on it also scores on how much of the question's own vocabulary appears in its
    title and body — which is where article numbers, product codes and phrasing
    the keyword list does not carry actually live. The two modes therefore rank
    the corpus differently, and a question can reach a passage under one and not
    the other.

    Returns up to *top_k* passages with at least one hit, ordered by descending
    hits. Never mutates the knowledge base.
    """
    low = normalized_question.lower()
    terms = _question_terms(low) if hybrid_search else set()
    scored: List[Dict[str, Any]] = []
    for passage in _KNOWLEDGE_BASE:
        hits = sum(1 for kw in passage["keywords"] if kw.lower() in low)
        if hybrid_search:
            body = f"{passage['title']} {passage['text']}".lower()
            hits += sum(1 for term in terms if term in body)
        if hits >= 1:
            scored.append(
                {
                    "id": passage["id"],
                    "title": passage["title"],
                    "text": passage["text"],
                    "source": passage["source"],
                    "hits": hits,
                }
            )
    scored.sort(key=lambda p: p["hits"], reverse=True)
    return scored[:top_k]


class RetrieveNode(FunctionNode):
    """Retrieve candidate compliance passages for the question.

    Inner node: ANONYMOUS trust (see the module comment).

    Input state keys:
        qa_query:         str  — JSON-serialised {question, normalized, topics}
        runtime_settings: str  — JSON declared settings (top_k, hybrid_search)

    Output state keys (partial dict):
        retrieved_passages: str  — JSON-serialised candidate passage list
        retrieved_count:    int
        status:             str
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        qa_query: Dict[str, Any] = from_json(state.get("qa_query"), {}) or {}
        normalized = str(qa_query.get("normalized", "")).strip()

        settings: Dict[str, Any] = from_json(state.get("runtime_settings"), {}) or {}
        declared_top_k = finite_in_range(settings.get("top_k"), *_TOP_K_BOUNDS)
        top_k = int(declared_top_k) if declared_top_k is not None else DEFAULT_TOP_K
        raw_hybrid = settings.get("hybrid_search")
        hybrid_search = raw_hybrid if isinstance(raw_hybrid, bool) else DEFAULT_HYBRID_SEARCH

        if not normalized:
            logger.warning("RetrieveNode: no question to retrieve against")
            emit_trace_event(
                "retrieve_empty_query",
                {"reason": "normalized_question_absent"},
                state,
            )
            return {
                "retrieved_passages": to_json([]),
                "retrieved_count": 0,
                "status": AgentStatus.SUCCESS.value,
            }

        passages = _retrieve(normalized, top_k, hybrid_search)

        logger.info(
            "RetrieveNode: query_len=%d top_k=%d hybrid=%s retrieved=%d",
            len(normalized),
            top_k,
            hybrid_search,
            len(passages),
        )
        emit_trace_event(
            "retrieve_complete",
            {
                "top_k": top_k,
                "hybrid_search": hybrid_search,
                "retrieved_count": len(passages),
                "passage_ids": [p["id"] for p in passages],
            },
            state,
        )

        return {
            "retrieved_passages": to_json(passages),
            "retrieved_count": len(passages),
            "status": AgentStatus.SUCCESS.value,
        }
