"""Offline boundary tests with real temporary files and no indexing/models."""

import ast
import copy
import inspect
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def module_from_source(name, path):
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / path)
    exec(compile((ROOT / path).read_text(encoding="utf-8-sig"), str(path), "exec"), module.__dict__)
    return module


class GuardrailTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="rag-guardrails-")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        config = types.ModuleType("config")
        # Execute only the new plain constants, never dotenv or application code.
        tree = ast.parse((ROOT / "config.py").read_text())
        constants = [n for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id.startswith("CLOUD_MAX_") for t in n.targets)]
        exec(compile(ast.Module(body=constants, type_ignores=[]), "limits", "exec"), config.__dict__)
        config.DEPLOYMENT_MODE = "cloud"
        gr = types.ModuleType("gradio")
        gr.skip = lambda: {"__type__": "update"}
        gr.update = lambda **kwargs: dict(kwargs, __type__="update")
        gr.Warning = Mock()
        class UserError(Exception):
            def __init__(self, message, **kwargs):
                super().__init__(message)
        gr.Error = UserError
        indexing = types.ModuleType("ingestion.indexing")
        indexing.index_files = Mock(side_effect=lambda files, kb, *rest: (kb, "indexed", [], gr.update()))
        indexing.format_indexed_docs_status = Mock(return_value="empty")
        self.context = patch.dict(sys.modules, {"config": config, "gradio": gr,
                                               "ingestion.indexing": indexing})
        self.context.start()
        self.addCleanup(self.context.stop)
        self.guard = module_from_source("deployment.guardrails", "deployment/guardrails.py")
        self.guard_patch = patch.dict(sys.modules, {"deployment.guardrails": self.guard})
        self.guard_patch.start()
        self.addCleanup(self.guard_patch.stop)
        self.upload = module_from_source("isolated_upload", "ui/upload.py")
        self.helpers = module_from_source("isolated_helpers", "utils/helpers.py")
        self.gr = gr
        self.index = indexing.index_files
        self.reset = patch.object(self.upload, "reset_kb_state", side_effect=AssertionError("unexpected reset"))
        self.reset_mock = self.reset.start()
        self.addCleanup(self.reset.stop)

    def file(self, name, size=10):
        path = self.folder / name
        with path.open("wb") as stream:
            stream.write(name.encode()[:size])
            stream.truncate(size)
        return str(path)

    def kb(self, paths=(), mode="multi"):
        return {"mode": mode, "documents": {
            str(i): {"file_path": p, "fingerprint": self.guard.file_fingerprint(p),
                     "chunk_size": 700, "chunk_overlap": 100}
            for i, p in enumerate(paths)}, "bm25_chunks": ["existing"], "vectordb": None}

    def validate(self, files, kb=None, mode="multi", previous=(), chunk_size=700):
        return self.guard.validate_upload(files, previous, kb or self.kb(), mode, chunk_size, 100)

    def rejected(self, files, kb=None):
        kb = kb or self.kb()
        before = copy.deepcopy(kb)
        result = self.upload.upload_wrapper(files, [], kb, "Multi-document", 700, 100)
        self.assertEqual(len(result), 10)
        self.assertIsInstance(result[2], str)
        for i, value in enumerate(result):
            if i != 2:
                self.assertEqual(value, self.gr.skip())
        self.assertNotIn(str(self.folder), result[2])
        self.assertNotIn("PRIVATE", result[2])
        self.assertEqual(kb, before)
        self.index.assert_not_called()
        self.reset_mock.assert_not_called()
        return result[2]

    def test_configured_limits(self):
        self.assertEqual((self.guard.CLOUD_MAX_DOCUMENTS, self.guard.CLOUD_MAX_FILE_SIZE_BYTES,
                          self.guard.CLOUD_MAX_TOTAL_UPLOAD_BYTES, self.guard.CLOUD_MAX_QUERY_CHARACTERS),
                         (3, 10 * 1024 * 1024, 20 * 1024 * 1024, 2000))

    def test_supported_suffixes_and_cloud_upload(self):
        paths = [self.file("a.PDF"), self.file("b.DoC"), self.file("c.docx")]
        self.assertIsNone(self.validate(paths))
        kb = self.kb()
        result = self.upload.upload_wrapper(paths, [], kb, "Multi-document", 700, 100)
        self.assertEqual(result[2], "indexed")
        self.index.assert_called_once_with(paths, kb, "multi", 700, 100)

    def test_file_size_boundary(self):
        limit = self.guard.CLOUD_MAX_FILE_SIZE_BYTES
        self.assertIsNone(self.validate([self.file("exact.pdf", limit)]))
        self.rejected([self.file("over.pdf", limit + 1)])

    def test_document_count_and_accumulation(self):
        old = [self.file("old1.pdf"), self.file("old2.pdf")]
        kb = self.kb(old)
        self.assertIsNone(self.validate([self.file("third.pdf")], kb))
        self.rejected([self.file("third.pdf"), self.file("fourth.pdf")], kb)

    def test_total_size_boundary(self):
        half = self.guard.CLOUD_MAX_FILE_SIZE_BYTES
        old = self.file("old.pdf", half)
        new = self.file("new.pdf", half)
        kb = self.kb([old])
        self.assertIsNone(self.validate([new], kb))
        self.rejected([new, self.file("extra.pdf", 1)], kb)

    def test_duplicates_follow_indexing_keys(self):
        path = self.file("duplicate.pdf", self.guard.CLOUD_MAX_FILE_SIZE_BYTES)
        kb = self.kb([path])
        self.assertIsNone(self.validate([path, path, path, path], kb))
        # Re-indexing with different chunk settings counts as another document.
        self.assertIsNone(self.validate([path], kb, chunk_size=600))
        other = self.file("other.pdf", self.guard.CLOUD_MAX_FILE_SIZE_BYTES)
        self.assertIsNotNone(self.validate([path, other], kb, chunk_size=600))

    def test_single_replaces_and_mode_switch_does_not_accumulate(self):
        kb = self.kb([self.file("old.pdf", self.guard.CLOUD_MAX_FILE_SIZE_BYTES)])
        new = self.file("new.pdf", self.guard.CLOUD_MAX_FILE_SIZE_BYTES)
        self.assertIsNone(self.validate([new], kb, mode="single"))
        kb["mode"] = "single"
        self.assertIsNone(self.validate([new], kb))

    def test_rejection_precedes_uploader_removal_reset(self):
        old = self.file("old.pdf")
        kb = self.kb([old])
        before = copy.deepcopy(kb)
        result = self.upload.upload_wrapper([self.file("PRIVATE.exe")],
                    [self.guard.file_fingerprint(old)], kb, "Multi-document", 700, 100)
        self.assertIn("PDF", result[2])
        self.reset_mock.assert_not_called()
        self.index.assert_not_called()
        self.assertEqual(kb, before)

    def test_bad_temporary_files(self):
        self.rejected([object()])
        self.rejected([str(self.folder / "PRIVATE-missing.pdf")])
        directory = self.folder / "directory.pdf"
        directory.mkdir()
        self.rejected([str(directory)])
        valid = self.file("PRIVATE.pdf")
        with patch("builtins.open", side_effect=PermissionError("PRIVATE /home/person/file")):
            self.rejected([valid])

    def test_local_upload_bypasses_cloud_checks(self):
        self.guard.DEPLOYMENT_MODE = "local"
        paths = [str(self.folder / "PRIVATE.exe")] * 4
        kb = self.kb()
        result = self.upload.upload_wrapper(paths, [], kb, "Multi-document", 700, 100)
        self.assertEqual(result[2], "indexed")
        self.index.assert_called_once_with(paths, kb, "multi", 700, 100)

    def test_query_boundaries_and_downstream_skip(self):
        history = [{"role": "assistant", "content": "existing answer"}]
        original = copy.deepcopy(history)
        for length in (2000, 2001):
            downstream = Mock(return_value="called")
            guarded = self.guard.guard_chat(downstream)
            display, pending, textbox = self.helpers.prepare_submission("  " + "q" * length + "  ", history)
            result = guarded(pending, history, self.kb())
            if length == 2000:
                self.assertEqual(pending, "q" * length)
                downstream.assert_called_once()
            else:
                self.assertEqual((display, pending, textbox), (history, "", ""))
                self.assertEqual(result, tuple(self.gr.skip() for _ in range(5)))
                downstream.assert_not_called()
                self.gr.Warning.assert_called_once()
        self.assertEqual(history, original)

    def test_local_long_query_and_empty_handling(self):
        self.guard.DEPLOYMENT_MODE = "local"
        history = []
        self.assertEqual(self.helpers.prepare_submission(" " * 5, history), ([], "", ""))
        self.assertEqual(self.helpers.prepare_submission("q" * 2001, history)[1], "q" * 2001)
        downstream = Mock()
        self.guard.guard_chat(downstream)("", history, self.kb())
        downstream.assert_called_once()
        self.gr.Warning.assert_not_called()

    def test_direct_cloud_question_callback_is_guarded(self):
        downstream = Mock()
        guarded = self.guard.guard_chat(downstream)
        result = guarded(user_query="q" * 2001, chat_history=[], kb_state=self.kb())
        self.assertEqual(result, tuple(self.gr.skip() for _ in range(5)))
        downstream.assert_not_called()
        self.gr.Warning.assert_called_once()

    def test_valid_removal_keeps_existing_reset_semantics(self):
        old, remaining = self.file("old.pdf"), self.file("remaining.pdf")
        kb = self.kb([old, remaining])
        self.reset_mock.side_effect = None
        self.reset_mock.return_value = self.kb()
        result = self.upload.upload_wrapper([remaining], [self.guard.file_fingerprint(old),
                    self.guard.file_fingerprint(remaining)], kb, "Multi-document", 700, 100)
        self.reset_mock.assert_called_once_with(kb, mode="multi")
        self.index.assert_not_called()
        self.assertEqual(result[8], [])

    def test_signatures_and_both_submission_chains(self):
        self.assertEqual(list(inspect.signature(self.upload.upload_wrapper).parameters),
                         ["files", "previous_files", "kb_state", "mode_label", "chunk_size", "chunk_overlap"])
        self.assertEqual(list(inspect.signature(self.helpers.prepare_submission).parameters),
                         ["user_query", "chat_history"])
        def callback(user_query, chat_history, kb_state):
            pass
        self.assertEqual(inspect.signature(self.guard.guard_chat(callback)), inspect.signature(callback))
        tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8-sig"))
        self.assertTrue(any(isinstance(n, ast.Assign) and ast.unparse(n) == "chat = guard_chat(chat)"
                            for n in tree.body))
        submissions = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and n.args
                       and isinstance(n.args[0], ast.Name) and n.args[0].id == "prepare_submission"]
        chats = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and n.args
                 and not (isinstance(n.func, ast.Name) and n.func.id == "guard_chat")
                 and isinstance(n.args[0], ast.Name) and n.args[0].id == "chat"]
        self.assertEqual((len(submissions), len(chats)), (2, 2))

    def test_cloud_session_allowance(self):
        count = 0
        for expected in range(1, 6):
            count = self.guard.consume_query_allowance(count)
            self.assertEqual(count, expected)
        for _ in range(3):
            with self.assertRaises(self.guard.QueryLimitError):
                self.guard.consume_query_allowance(count)
            self.assertEqual(count, 5)

    def test_invalid_session_counts_fail_closed(self):
        for count in (None, True, False, -1, 6, 1.0, "1", {}, float("nan")):
            with self.subTest(count=count), self.assertRaises(self.guard.QueryLimitError):
                self.guard.consume_query_allowance(count)

    def test_session_validation_precedes_consumption(self):
        for query in ("", "   ", "q" * 2001):
            with patch.object(self.guard, "consume_query_allowance") as consume:
                result = self.helpers.prepare_session_submission(query, [], 4)
                self.assertEqual(result[1:], ("", "", 4))
                consume.assert_not_called()

    def test_sixth_question_and_failure_accounting(self):
        count = 0
        downstream = Mock(side_effect=RuntimeError("provider failure"))
        for _ in range(5):
            _, pending, _, count = self.helpers.prepare_session_submission("valid", [], count)
            with self.assertRaises(RuntimeError):
                downstream(pending)
        self.assertEqual(count, 5)
        for _ in range(2):
            with self.assertRaisesRegex(self.gr.Error, "This public demo allows up to 5 questions per session."):
                _, pending, _, count = self.helpers.prepare_session_submission("valid", [], count)
                downstream(pending)
        self.assertEqual(downstream.call_count, 5)
        self.assertEqual(count, 5)

    def test_local_session_allowance_is_unused(self):
        self.guard.DEPLOYMENT_MODE = "local"
        for count in (0, 5, 100, None):
            for _ in range(7):
                result = self.helpers.prepare_session_submission("q" * 2001, [], count)
                self.assertEqual(result[1], "q" * 2001)
                self.assertEqual(result[3], count)


if __name__ == "__main__":
    unittest.main()
