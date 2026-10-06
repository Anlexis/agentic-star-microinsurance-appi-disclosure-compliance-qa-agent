"""AgentCore Platform v1.0"""

# Provenance for an answer grounded in the corpus shipped inside this template.
#
# Every one of these agents answers from a table defined in its own node module, and says so
# only in a docstring. The reader of the answer sees a citation -- an APPI article, an EDINET
# document id, a policy clause -- with nothing telling them the corpus travels with the
# template rather than being a live query against the system of record.
#
# The platform review round rated that the most serious class it found: a technical limitation
# rendered as authoritative information. The template it flagged hardest asserted "based on
# provided model inventory" over two models hardcoded in a node. These agents do not assert
# anything false; they simply do not deny it, which is the same failure one step milder.
#
# So the disclosure goes where the reader is, not in the source.

from typing import Any, Dict

# Keys a pipeline may use to record when its bundled corpus was captured.
_SNAPSHOT_KEYS = (
    "kb_snapshot",
    "kb_snapshot_date",
    "corpus_snapshot",
    "snapshot_date",
    "kb_version",
    "knowledge_base_version",
)


def _snapshot_from(state: Dict[str, Any]) -> str:
    for key in _SNAPSHOT_KEYS:
        value = state.get(key)
        if value:
            return str(value)
    return ""


def source_label(state: Dict[str, Any]) -> str:
    """The provenance footer, appended to the full answer."""
    snapshot = _snapshot_from(state)
    when = f"snapshot {snapshot}" if snapshot else "no snapshot date recorded"
    return (
        "\n\n---\n"
        "**Source of this answer.** Grounded in a reference corpus bundled with this "
        f"template ({when}), not a live query against a system of record. Treat it as a "
        "starting point for a decision rather than the decision: verify against the "
        "authoritative source before acting on it.\n"
        "本回答は本テンプレートに同梱された参照コーパスに基づくものであり、記録システムへの"
        "照会結果ではありません。実行前に必ず正本をご確認ください。"
    )


def source_note(state: Dict[str, Any]) -> str:
    """One line, for a field that carries a short result rather than a full answer."""
    snapshot = _snapshot_from(state)
    when = f", snapshot {snapshot}" if snapshot else ""
    return (
        f" [Source: reference corpus bundled with this template{when}; not a system of record -- verify before acting.]"
    )
