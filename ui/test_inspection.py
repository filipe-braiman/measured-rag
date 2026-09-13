"""Offline inspection formatting and debug presentation contracts."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from deployment import test_guardrails as fixtures


class InspectionTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.GuardrailTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.views = fixtures.module_from_source("isolated_views", "ui/view_toggles.py")
        self.reset = fixtures.module_from_source("isolated_reset", "ui/reset.py")

    def test_refresh_preserves_complete_collection_and_format(self):
        chunks = [SimpleNamespace(page_content=f"Passage {i}", metadata={
            "source_name": f"paper-{i}.pdf", "chunk_id": f"doc_{i}",
            "page": i, "section_header": "Methods", "doc_type": "pdf",
            "source": "/private/unused-metadata.pdf",
        }) for i in (1, 2)]
        result = self.views.render_indexed_chunks(chunks)
        self.assertEqual(result, self.views.toggle_chunks(True, chunks))
        self.assertTrue(result["visible"])
        self.assertIn("Passage 1", result["value"])
        self.assertIn("Passage 2", result["value"])
        self.assertNotIn("/private/", result["value"])
        self.assertEqual(chunks[0].metadata["source"], "/private/unused-metadata.pdf")

    def test_empty_mode_change_and_chat_reset(self):
        from core.state import new_kb_state
        kb = new_kb_state()
        kb["bm25_chunks"] = [SimpleNamespace(page_content="Retained", metadata={})]
        reset = self.reset.reset_chat_only(kb)
        self.assertIn("Retained", self.views.render_indexed_chunks(reset[2])["value"])
        self.assertEqual(reset[4:6], ("", ""))
        switched = self.reset.on_mode_change("Multi-document", kb)
        self.assertEqual(self.views.render_indexed_chunks(switched[2])["value"], "")
        self.assertEqual(switched[4:6], ("", ""))
        self.assertEqual(self.views.render_indexed_chunks([])["value"], "")

    def test_debug_flag_is_unused_and_payload_is_unconditional(self):
        tree = ast.parse((fixtures.ROOT / "chat/service.py").read_text(encoding="utf-8"))
        chat = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "chat")
        self.assertIn("debug_enabled", [a.arg for a in chat.args.args])
        self.assertFalse(any(isinstance(n, ast.Name) and n.id == "debug_enabled"
                             for n in ast.walk(chat)))
        assignments = [n for n in chat.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "debug_text" for t in n.targets)]
        self.assertEqual(len(assignments), 1)
        self.assertEqual(ast.unparse(assignments[0].value.func), "build_debug_info")
        self.assertEqual(ast.unparse(chat.body[-1].value.elts[3]), "debug_text")


if __name__ == "__main__":
    unittest.main()
