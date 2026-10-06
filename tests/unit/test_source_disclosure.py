"""The provenance footer must survive, and must be reachable from the node."""

import inspect

from src.services.source_disclosure import source_label, source_note


class TestTheFooterSaysWhereTheAnswerCameFrom:
    def test_it_names_the_bundled_corpus_in_both_languages(self):
        text = source_label({})
        assert "bundled with this template" in text
        assert "本テンプレートに同梱" in text

    def test_it_says_so_even_with_no_snapshot_recorded(self):
        assert "no snapshot date recorded" in source_label({})

    def test_a_recorded_snapshot_reaches_the_reader(self):
        assert "2026-01-31" in source_label({"kb_snapshot": "2026-01-31"})

    def test_the_short_form_carries_the_same_warning(self):
        assert "not a system of record" in source_note({})


class TestTheNodeActuallyAppendsIt:
    """Asserting the module alone would pass while nothing called it -- which is the state
    this batch shipped to the registry without noticing."""

    def test_the_node_calls_the_footer(self):
        from src.nodes import output_format_node as node_module

        assert "source_label(" in inspect.getsource(node_module)
