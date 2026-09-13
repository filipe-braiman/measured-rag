"""Offline telemetry contract tests using isolated temporary repository roots."""

import ast
from contextlib import redirect_stdout
from datetime import datetime
import inspect
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "telemetry" / "logger.py").read_text(encoding="utf-8-sig")
PARAMETERS = """kb_mode active_doc_names conversation_id query_id original_query
rewritten_query retrieval_route fusion_alpha retrieval_top_n rerank_candidate_n
rerank_top_k fusion_threshold retrieval_relevance retrieval_relevance_score
retrieval_explanation retrieval_eval_latency groundedness_label
groundedness_explanation groundedness_eval_latency answer raw_docs reranked_docs
reranked_dists rerank_scores retrieval_latency dense_latency bm25_latency
fusion_latency rerank_latency tokens llm_latency""".split()


class FrozenDatetime:
    @staticmethod
    def now():
        return datetime(2026, 1, 2, 3, 4, 5)


def load_logger(root, mode):
    # Execute the real module initialization with a synthetic __file__ and
    # settings module: no dotenv access, app import, or project log writes.
    module = types.ModuleType("isolated_telemetry_logger")
    module.__file__ = str(root / "telemetry" / "logger.py")
    config = types.ModuleType("config")
    config.DEPLOYMENT_MODE = mode
    with patch.dict("sys.modules", {"config": config}):
        exec(compile(SOURCE, module.__file__, "exec"), module.__dict__)
    module.datetime = FrozenDatetime
    return module


def sample_arguments():
    doc = types.SimpleNamespace(
        page_content="SENTINEL_CONTENT /home/private-person/private.txt",
        metadata={"doc_id": "SENTINEL_DOCUMENT_ID", "chunk_id": "SENTINEL_CHUNK_ID",
                  "source_name": r"C:\Users\private-person\SENTINEL_FILENAME.docx",
                  "section_header": "SENTINEL_SECTION", "page": 2,
                  "file_type": "docx", "doc_type": "section", "fusion_score": 0.7},
    )
    return dict(
        kb_mode="single", active_doc_names=[doc.metadata["source_name"]],
        conversation_id="SENTINEL_CONVERSATION", query_id="SENTINEL_QUERY_ID",
        original_query="SENTINEL_QUESTION", rewritten_query="SENTINEL_REWRITE",
        retrieval_route="balanced", fusion_alpha=0.65, retrieval_top_n=10,
        rerank_candidate_n=5, rerank_top_k=3, fusion_threshold=0.1,
        retrieval_relevance="high", retrieval_relevance_score=0.9,
        retrieval_explanation="SENTINEL_RELEVANCE_EXPLANATION", retrieval_eval_latency=0.2,
        groundedness_label="grounded", groundedness_explanation="SENTINEL_GROUNDEDNESS_EXPLANATION",
        groundedness_eval_latency=0.3, answer="SENTINEL_ANSWER",
        raw_docs=[(doc, 0.25, 0.7)], reranked_docs=[doc], reranked_dists=[0.25],
        rerank_scores=[0.8], retrieval_latency=1.0, dense_latency=0.4,
        bm25_latency=0.1, fusion_latency=0.2, rerank_latency=0.3, tokens=123, llm_latency=2.0,
    )


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="rag-telemetry-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_local_schema_values_and_append(self):
        logger = load_logger(self.root, "local")
        path = self.root / "answer_logs" / "rag_eval_log.jsonl"
        self.assertEqual(logger.EVAL_LOG_PATH, path)
        self.assertTrue(path.parent.is_dir())
        args = sample_arguments()
        with redirect_stdout(io.StringIO()) as output:
            logger.log_evaluation_entry(**args)
            first = path.read_bytes()
            logger.log_evaluation_entry(**args)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(path.read_bytes(), first + first)
        raw_source = dict(rank=1, doc_id="SENTINEL_DOCUMENT_ID",
                          file_name=args["active_doc_names"][0], chunk_id="SENTINEL_CHUNK_ID",
                          page=2, distance=0.25, similarity=0.8, fusion_score=0.7,
                          content_preview=args["raw_docs"][0][0].page_content,
                          file_type="docx", doc_type="section", section_header="SENTINEL_SECTION")
        expected = {
            "timestamp": "2026-01-02T03:04:05", "conversation_id": args["conversation_id"],
            "query_id": args["query_id"], "kb_mode": "single",
            "active_doc_names": args["active_doc_names"],
            "query": {"original": args["original_query"], "rewritten": args["rewritten_query"]},
            "retrieval": {
                "route": "balanced", "fusion_alpha": 0.65, "retrieval_top_n": 10,
                "rerank_candidate_n": 5, "rerank_top_k": 3, "fusion_threshold": 0.1,
                "latency": {"total_seconds": 1.0, "dense_seconds": 0.4,
                            "bm25_seconds": 0.1, "fusion_seconds": 0.2, "rerank_seconds": 0.3},
                "num_fusion_survivors": 1, "num_reranker_candidates": 1,
                "raw_retrieval": [raw_source],
                "final_sources": [dict(raw_source, rerank_score=0.8)], "rerank_margin": None,
                "retrieval_relevance": "high", "relevance_score": 0.9,
                "relevance_explanation": args["retrieval_explanation"],
                "retrieval_eval_latency": 0.2, "num_final_sources": 1,
            },
            "generation": {"answer": args["answer"], "groundedness_label": "grounded",
                           "groundedness_explanation": args["groundedness_explanation"],
                           "groundedness_eval_latency": 0.3},
            "llm_metrics": {"total_tokens": 123, "latency_seconds": 2.0},
        }
        self.assertEqual(json.loads(first), expected)

    def test_cloud_exact_schema_and_privacy(self):
        logger = load_logger(self.root, "cloud")
        args = sample_arguments()
        output = io.StringIO()
        with redirect_stdout(output), patch.object(output, "flush", wraps=output.flush) as flush:
            logger.log_evaluation_entry(**args)
            flush.assert_called_once_with()
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0]), {
            "event": "rag_request_completed", "timestamp": "2026-01-02T03:04:05",
            "kb_mode": "single", "retrieval_route": "balanced", "fusion_alpha": 0.65,
            "retrieval_top_n": 10, "rerank_candidate_n": 5, "rerank_top_k": 3,
            "fusion_threshold": 0.1, "num_fusion_survivors": 1,
            "num_reranker_candidates": 1, "num_final_sources": 1,
            "retrieval_relevance": "high", "relevance_score": 0.9,
            "groundedness_label": "grounded", "retrieval_latency": 1.0,
            "dense_latency": 0.4, "bm25_latency": 0.1, "fusion_latency": 0.2,
            "rerank_latency": 0.3, "retrieval_eval_latency": 0.2,
            "groundedness_eval_latency": 0.3, "llm_latency": 2.0, "total_tokens": 123,
        })
        for forbidden in ("SENTINEL", "private-person", "C:", "/home/", "hash", "preview"):
            self.assertNotIn(forbidden, output.getvalue())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_cloud_never_inspects_sensitive_arguments(self):
        logger = load_logger(self.root, "cloud")
        class Unreadable:
            def __getattribute__(self, name):
                raise AssertionError("SENTINEL_PRIVATE")
            def __str__(self):
                raise AssertionError("SENTINEL_PRIVATE")
            def __len__(self):
                raise AssertionError("SENTINEL_PRIVATE")
        args = sample_arguments()
        for key in ("active_doc_names", "conversation_id", "query_id", "original_query",
                    "rewritten_query", "answer", "retrieval_explanation",
                    "groundedness_explanation", "reranked_dists", "rerank_scores"):
            args[key] = Unreadable()
        args["raw_docs"] = [Unreadable()]
        args["reranked_docs"] = [Unreadable()]
        with redirect_stdout(io.StringIO()) as output:
            logger.log_evaluation_entry(**args)
        self.assertEqual(json.loads(output.getvalue())["event"], "rag_request_completed")

    def test_cloud_failures_are_generic_and_nonfatal(self):
        logger = load_logger(self.root, "cloud")
        for value in ("SENTINEL_PRIVATE", float("nan"), float("inf"), object()):
            args = sample_arguments()
            args["fusion_alpha"] = value
            with redirect_stdout(io.StringIO()) as output:
                self.assertIsNone(logger.log_evaluation_entry(**args))
            self.assertEqual(json.loads(output.getvalue()), {"event": "rag_telemetry_error"})
        with patch.object(logger, "_emit_cloud_event", side_effect=OSError("SENTINEL_PRIVATE")):
            self.assertIsNone(logger.log_evaluation_entry(**sample_arguments()))
        for method in ("write", "flush"):
            with redirect_stdout(io.StringIO()) as output:
                with patch.object(output, method, side_effect=OSError("SENTINEL_PRIVATE")):
                    self.assertIsNone(logger.log_evaluation_entry(**sample_arguments()))
                self.assertNotIn("SENTINEL", output.getvalue())
        with redirect_stdout(io.StringIO()) as output:
            with patch.object(logger, "_emit_cloud_event",
                              side_effect=[ValueError("SENTINEL_PRIVATE"), None]) as emit:
                logger.log_evaluation_entry(**sample_arguments())
            self.assertEqual(emit.call_args.args, ({"event": "rag_telemetry_error"},))
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_cloud_labels_cannot_carry_arbitrary_text(self):
        logger = load_logger(self.root, "cloud")
        args = sample_arguments()
        for key in ("kb_mode", "retrieval_route", "retrieval_relevance", "groundedness_label"):
            args[key] = "SENTINEL_PRIVATE"
        args["tokens"] = args["retrieval_relevance_score"] = None
        with redirect_stdout(io.StringIO()) as output:
            logger.log_evaluation_entry(**args)
        event = json.loads(output.getvalue())
        self.assertNotIn("SENTINEL", output.getvalue())
        self.assertEqual(event["retrieval_relevance"], "unknown")
        self.assertIsNone(event["total_tokens"])
        self.assertIsNone(event["relevance_score"])

    def test_unsupported_mode_and_signature(self):
        for mode in ("", "CLOUD", "SENTINEL_PRIVATE", None):
            with redirect_stdout(io.StringIO()) as output:
                with self.assertRaisesRegex(ValueError, '^DEPLOYMENT_MODE must be exactly "local" or "cloud".$'):
                    load_logger(self.root, mode)
            self.assertEqual(output.getvalue(), "")
            self.assertEqual(list(self.root.iterdir()), [])
        logger = load_logger(self.root, "cloud")
        self.assertEqual(list(inspect.signature(logger.log_evaluation_entry).parameters), PARAMETERS)
        tree = ast.parse((ROOT / "chat" / "service.py").read_text(encoding="utf-8-sig"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "log_evaluation_entry"]
        self.assertEqual(len(calls), 1)
        self.assertEqual({kw.arg for kw in calls[0].keywords}, set(PARAMETERS))

    def test_docker_cleanup_and_test_exclusion(self):
        docker = (ROOT / "Dockerfile").read_text()
        self.assertNotIn("/app/answer_logs", docker)
        self.assertIn("/runtime/gradio /runtime/cache", docker)
        self.assertIn("COPY telemetry/logger.py /app/telemetry/", docker)
        self.assertNotIn("telemetry/test_logger.py", docker)
        self.assertNotIn("COPY telemetry/ ", docker)
        for name in (".dockerignore", ".gcloudignore"):
            rules = (ROOT / name).read_text().splitlines()
            self.assertIn("**", rules)
            self.assertNotIn("!telemetry/test_logger.py", rules)


if __name__ == "__main__":
    unittest.main()
