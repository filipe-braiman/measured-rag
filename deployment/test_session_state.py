"""Real Gradio Blocks/state verification with application services stubbed."""

import ast
import os
import sys
import unittest
from unittest.mock import patch

from deployment import test_guardrails as fixtures

ROOT = fixtures.ROOT


class SessionStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_blocks_and_server_side_counter(self):
        environment = patch.dict(os.environ, {"GRADIO_ANALYTICS_ENABLED": "False"})
        environment.start()
        self.addCleanup(environment.stop)
        # Keep this construction/state test offline, including optional library background checks. No model/provider services are imported below.
        network = patch("socket.socket.connect", side_effect=OSError("Offline test"))
        network.start()
        self.addCleanup(network.stop)
        import gradio as gr
        from gradio.state_holder import SessionState

        fixture = fixtures.GuardrailTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        # The entire real Blocks declaration is executed. Only application services are replaced; no app import, model initialization or launch.
        tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8-sig"))
        blocks = next(n for n in tree.body if isinstance(n, ast.With))
        queue_setup = [n for n in tree.body if isinstance(n, ast.If)
                       and ast.unparse(n.test) == "DEPLOYMENT_MODE == 'cloud'"]
        self.assertEqual(len(queue_setup), 1)
        def callback(*args, **kwargs):
            return None
        namespace = {alias.asname or alias.name: callback for n in tree.body
                     if isinstance(n, ast.ImportFrom) for alias in n.names}
        namespace.update(gr=gr, new_kb_state=lambda **kw: fixture.kb(),
                         new_conversation_id=lambda: "test-conversation",
                         prepare_submission=fixture.helpers.prepare_session_submission,
                         chat=lambda *args: None, upload_wrapper=lambda *args: None)
        # Presentation constants must remain real for Blocks construction.
        from ui import branding
        namespace.update({name: getattr(branding, name) for name in
                          ("ABOUT_HTML", "DEMO_NOTICE_HTML", "EVIDENCE_NOTE_HTML", "HEADER_HTML", "STYLESHEET_PATH")})
        from ui.view_toggles import render_indexed_chunks
        namespace["render_indexed_chunks"] = render_indexed_chunks
        with patch.dict(sys.modules, {"gradio": gr}):
            for mode in ("cloud", "local"):
                fixture.guard.DEPLOYMENT_MODE = mode
                namespace["DEPLOYMENT_MODE"] = mode
                exec(compile(ast.Module(body=[blocks, *queue_setup], type_ignores=[]),
                             "app-blocks", "exec"), namespace)
                app = namespace["app"]
                notices = [c for c in app.blocks.values() if isinstance(c, gr.HTML)
                           and "portfolio-demo-notice" in c.value]
                self.assertEqual(len(notices), 1 if mode == "cloud" else 0)
                panels = [c for c in app.blocks.values() if isinstance(c, gr.Accordion)
                          and c.label in {"Indexed Chunks", "Retrieved Sources", "Debug Last Answer"}]
                self.assertEqual(len(panels), 3)
                self.assertTrue(all(c.open for c in panels))
                for name in ("chunks_view", "sources", "debug_box"):
                    self.assertTrue(namespace[name].visible)
                    self.assertFalse(namespace[name].interactive)
                self.assertNotIn("show_chunks", namespace)
                self.assertNotIn("show_sources", namespace)
                self.assertIsInstance(namespace["debug_mode"], gr.State)
                self.assertIs(namespace["debug_mode"].value, False)
                refreshes = [f for f in app.fns.values() if f.fn is render_indexed_chunks]
                self.assertEqual(len(refreshes), 3)  # upload, mode change, chat reset
                for refresh in refreshes:
                    self.assertEqual(refresh.inputs, [namespace["state_chunks"]])
                    self.assertEqual(refresh.outputs, [namespace["chunks_view"]])
                self.assertIsInstance(namespace["selected_docs"], gr.CheckboxGroup)
                self.assertEqual(namespace["selected_docs"].value, [])
                self.assertEqual(namespace["selected_docs"].type, "value")
                count = namespace["state_query_count"]
                self.assertEqual(count.value, 0)
                admissions = [f for f in app.fns.values()
                              if f.fn is fixture.helpers.prepare_session_submission]
                self.assertEqual(len(admissions), 2)
                for admission in admissions:
                    self.assertIs(admission.inputs[-1], count)
                    self.assertIs(admission.outputs[-1], count)
                    self.assertEqual(admission.concurrency_limit, 1)
                    self.assertTrue(admission.queue)
                self.assertEqual(admissions[0].concurrency_id, admissions[1].concurrency_id)
                # No upload/reset/mode-change callback may overwrite allowance.
                writers = [f for f in app.fns.values() if count in f.outputs]
                self.assertEqual(writers, admissions)
                chats = [f for f in app.fns.values() if f.fn is namespace["chat"]]
                uploads = [f for f in app.fns.values() if f.fn is namespace["upload_wrapper"]]
                self.assertEqual((len(chats), len(uploads)), (2, 1))
                for event in chats:
                    self.assertIs(event.inputs[10], namespace["selected_docs"])
                    self.assertIs(event.inputs[8], namespace["debug_mode"])
                    self.assertEqual(event.outputs[2:4], [namespace["sources"], namespace["debug_box"]])
                for admission in admissions:
                    children = [f for f in app.fns.values() if f.trigger_after == admission._id]
                    self.assertEqual(len(children), 1)
                    self.assertIn(children[0], chats)
                    self.assertEqual(children[0].trigger_only_on_success, mode == "cloud")
                if mode == "cloud":
                    self.assertEqual(admissions[0].concurrency_id, "query-admission")
                    self.assertEqual(app._queue.max_size, 20)
                    self.assertEqual(app._queue.default_concurrency_limit, 1)
                    for event in chats + uploads:
                        self.assertEqual(event.concurrency_id, "cloud-heavy-work")
                        self.assertEqual(event.concurrency_limit, 1)
                        self.assertTrue(event.queue)
                    heavy = [f for f in app.fns.values() if f.concurrency_id == "cloud-heavy-work"]
                    self.assertEqual(len(heavy), 3)
                    self.assertEqual(set(heavy), set(chats + uploads))
                    self.assertNotEqual(admissions[0].concurrency_id, heavy[0].concurrency_id)
                    self.assert_scheduler_serialization(app, chats, uploads)
                    state = SessionState(app)
                    async def exercise():
                        # Simulate stale browser counter payloads while retaining the same server session, alternating Send and Enter.
                        for index in range(5):
                            await app.process_api(admissions[index % 2],
                                                  ["question", [], 0], state=state)
                            self.assertEqual(state[count._id], index + 1)
                        for index in range(2):
                            with self.assertRaises(gr.Error):
                                await app.process_api(admissions[index],
                                                      ["question", [], 0], state=state)
                            self.assertEqual(state[count._id], 5)
                    await exercise()
                else:
                    self.assertIsNone(app._queue.max_size)
                    for event in chats + uploads:
                        self.assertNotIn(event.concurrency_id,
                                         {"cloud-heavy-work", "query-admission"})
                        self.assertEqual(event.concurrency_limit, "default")
                app.close()

    def assert_scheduler_serialization(self, app, chats, uploads):
        """Exercise Gradio's real dispatcher with controlled active counts.

        No sleeps or services: simulate the active-count bookkeeping normally
        performed by start_processing/process_events around a running callback.
        """
        from gradio.queueing import Event
        queue = app._queue
        for fn in chats + uploads:
            queue.create_event_queue_for_fn(fn)
        heavy = queue.event_queue_per_concurrency_id["cloud-heavy-work"]
        self.assertEqual(heavy.concurrency_limit, 1)
        for running in chats + uploads:
            for waiting in (uploads + chats, chats + uploads):
                with self.subTest(running=running._id, order=[fn._id for fn in waiting]):
                    # Simulate any of the three callbacks already executing.
                    heavy.current_concurrency = 1
                    events = [Event(f"tab-{index}", fn, None, None)
                              for index, fn in enumerate(waiting)]
                    heavy.queue.extend(events)
                    self.assertIsNone(queue.get_events())
                    self.assertEqual(heavy.queue, events)
                    for index, expected in enumerate(events):
                        # Completion releases the group; the next queued event must be selected in arrival order, with no jumping.
                        heavy.current_concurrency = 0
                        selected, batch, group = queue.get_events()
                        self.assertEqual(group, "cloud-heavy-work")
                        self.assertEqual(selected, [expected])
                        self.assertFalse(batch)
                        heavy.current_concurrency = 1
                        self.assertIsNone(queue.get_events())
                        self.assertEqual(heavy.queue, events[index + 1:])
                    heavy.current_concurrency = 0
                self.assertIsNone(queue.get_events())


if __name__ == "__main__":
    unittest.main()
