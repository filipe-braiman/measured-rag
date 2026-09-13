"""Offline document-selection contracts using real Gradio processing.

Execute isolated source functions, never app.py or provider/model imports.
Only indexing infrastructure (embedding, chunk loading, vector store) is stubbed.
"""

import ast
import os
from pathlib import Path
import time
import unittest
from unittest.mock import Mock, patch

from core.state import new_kb_state, reset_kb_state
from deployment import test_guardrails as fixtures


class SelectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"GRADIO_ANALYTICS_ENABLED": "False"})
        environment.start()
        self.addCleanup(environment.stop)
        network = patch("socket.socket.connect", side_effect=OSError("Offline test"))
        network.start()
        self.addCleanup(network.stop)
        import gradio as gr
        self.gr = gr
        fixture = fixtures.GuardrailTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        fixture.guard.DEPLOYMENT_MODE = "local"
        fixture.reset_mock.side_effect = reset_kb_state
        self.reset = fixtures.module_from_source("isolated_reset", "ui/reset.py")

        # Exercise real index_files branches with no files/models/vector DB.
        def metadata(file_path, chunks, chunk_size, chunk_overlap):
            name = Path(file_path).name
            return {"doc_id": name, "file_name": name, "file_type": "pdf",
                    "fingerprint": name, "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap, "chunk_count": 1}, chunks

        self.index_namespace = dict(
            gr=gr, os=os, time=time, new_kb_state=new_kb_state,
            reset_kb_state=reset_kb_state,
            ensure_vectordb=lambda kb: dict(kb, vectordb=Mock()),
            file_fingerprint=lambda path: Path(path).name,
            is_supported_file=lambda path: True,
            prepare_chunks_for_file=lambda **kwargs: ["offline chunk"],
            build_document_metadata=metadata,
            rebuild_bm25_index=lambda kb: kb,
        )
        tree = ast.parse((fixtures.ROOT / "ingestion/indexing.py").read_text(encoding="utf-8"))
        names = {"index_files", "format_indexed_docs_status",
                 "get_doc_name_to_id_map", "get_active_doc_ids"}
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=functions, type_ignores=[]), "indexing-contract", "exec"),
             self.index_namespace)
        fixture.index.side_effect = self.index_namespace["index_files"]

    async def test_selector_update_roundtrip_and_scope(self):
        gr = self.gr
        from gradio.state_holder import SessionState
        with gr.Blocks(analytics_enabled=False) as app:
            docs = gr.CheckboxGroup(choices=["a.pdf", "b.pdf"], value=["a.pdf"])
            scope = gr.Radio(choices=["all", "selected"], value="all")
            scope.change(self.reset.on_scope_change, inputs=scope, outputs=docs)
            update_payload = gr.JSON(visible=False)
            update_payload.change(lambda update: update, inputs=update_payload, outputs=docs)
        self.addCleanup(app.close)
        state = SessionState(app)
        for scope_value in ("selected", "all", "selected"):
            result = await app.process_api(0, [scope_value], state=state)
            self.assertEqual(result["data"][0]["visible"], scope_value == "selected")
            self.assertNotIn("value", result["data"][0])
        # Gradio rebuilds the server-side component without its value; the
        # browser retains its selection because the updates omit "value".
        for selected in ([], ["a.pdf"], ["b.pdf", "a.pdf"]):
            self.assertEqual(docs.preprocess(selected), selected)
            self.assertEqual(docs.postprocess(selected), selected)
        result = await app.process_api(1, [gr.update(choices=["a.pdf", "b.pdf", "c.pdf"], value=[])], state=state)
        self.assertEqual(result["data"][0]["value"], [])
        self.assertEqual(state.blocks_config.blocks[docs._id].choices,
                         [("a.pdf", "a.pdf"), ("b.pdf", "b.pdf"), ("c.pdf", "c.pdf")])
        kb = new_kb_state("multi")
        kb["documents"] = {"id-a": {"file_name": "a.pdf"}, "id-b": {"file_name": "b.pdf"}}
        active = self.index_namespace["get_active_doc_ids"]
        self.assertEqual(active(kb, "all", []), {"id-a", "id-b"})
        self.assertEqual(active(kb, "selected", docs.preprocess(["b.pdf"])), {"id-b"})
        self.assertEqual(active(kb, "selected", []), set())

    def test_mode_transitions_and_chat_reset(self):
        kb = new_kb_state()
        for label, mode in (("Multi-document", "multi"), ("Single-document", "single")):
            result = self.reset.on_mode_change(label, kb)
            self.assertEqual(len(result), 10)
            kb = result[0]
            self.assertEqual(kb["mode"], mode)
            self.assertEqual(result[8], self.gr.update(choices=[], value=[], visible=False))
            self.assertEqual(result[9], self.gr.update(value="all", visible=mode == "multi"))
            self.assertEqual(result[7]["file_count"], "multiple" if mode == "multi" else "single")
        chat_reset = self.reset.reset_chat_only(kb)
        self.assertEqual(len(chat_reset), 7)
        self.assertIs(chat_reset[0], kb)
        self.assertEqual(chat_reset[1], [])

    def test_initial_additional_duplicate_and_empty_uploads(self):
        fixture = self.fixture
        first = fixture.file("a.pdf")
        second = fixture.file("long-" + "document-" * 12 + ".pdf")
        kb, previous = new_kb_state("multi"), []
        for files, expected in (([first], [Path(first).name]),
                                ([first, second], [Path(first).name, Path(second).name]),
                                ([first, second], [Path(first).name, Path(second).name]),
                                ([], [])):
            result = fixture.upload.upload_wrapper(files, previous, kb, "Multi-document", 700, 100)
            self.assertEqual(len(result), 10)
            kb, previous = result[0], result[8]
            update = result[4]
            self.assertEqual(update, self.gr.update(choices=expected, value=[]))
            # Existing indexing resets selections on every accepted upload.
            component = self.gr.CheckboxGroup(**{k: v for k, v in update.items() if k != "__type__"})
            self.assertEqual(component.value, [])
            self.assertEqual(component.preprocess(expected), expected)


if __name__ == "__main__":
    unittest.main()
