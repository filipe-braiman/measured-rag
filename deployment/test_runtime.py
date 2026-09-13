"""Offline launch-contract tests; never import the interactive application."""

import ast
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from deployment.runtime import gradio_launch_options, social_metadata_app_kwargs


ROOT = Path(__file__).resolve().parents[1]


class LaunchOptionsTests(unittest.TestCase):
    def test_local_preserves_share_and_ignores_port(self):
        for share in (False, True):
            for port in (None, "8080", "malformed", ""):
                with self.subTest(share=share, port=port):
                    with patch.dict(os.environ, {} if port is None else {"PORT": port}, clear=True):
                        self.assertEqual(gradio_launch_options("local", share), {"share": share})

    def test_cloud_ports_and_sharing(self):
        for port in (1, 8080, 65535):
            for share in (False, True):
                with self.subTest(port=port, share=share):
                    with patch.dict(os.environ, {"PORT": str(port)}, clear=True):
                        self.assertEqual(gradio_launch_options("cloud", share), {
                            "share": False, "server_name": "0.0.0.0", "server_port": port,
                        })

    def test_cloud_rejects_invalid_port(self):
        for port in (None, "", " ", "abc", "8080.0", "0", "-1", "65536"):
            with self.subTest(port=port):
                with patch.dict(os.environ, {} if port is None else {"PORT": port}, clear=True):
                    with self.assertRaisesRegex(ValueError, "PORT"):
                        gradio_launch_options("cloud", False)

    def test_modes_are_exact(self):
        for mode in ("", "LOCAL", "Cloud", " cloud", "cloud ", "staging", None):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "DEPLOYMENT_MODE"):
                    gradio_launch_options(mode, False)

    def test_application_launch_guard(self):
        # Inspect the complete source, then execute only its real launch guard.
        # Model loading, dotenv, provider construction and Blocks are not run.
        tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8-sig"))
        launches = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == "launch"]
        self.assertEqual(len(launches), 1)
        guards = [node for node in tree.body if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "__name__ == '__main__'"]
        self.assertEqual(len(guards), 1)
        self.assertIn(launches[0], list(ast.walk(guards[0])))
        self.assertEqual(ast.unparse(launches[0].func), "app.launch")
        code = compile(ast.Module(body=guards, type_ignores=[]), "app-launch-guard", "exec")
        for name in ("app", "__main__"):
            for mode in ("local", "cloud"):
                with self.subTest(name=name, mode=mode):
                    application = Mock()
                    with patch.dict(os.environ, {"PORT": "8080"}, clear=True):
                        exec(code, {"__name__": name, "app": application,
                                    "DEPLOYMENT_MODE": mode, "SHARE": True,
                                    "FAVICON_PATH": ROOT / "assets/measured-rag-favicon-64.png",
                                    "social_metadata_app_kwargs": social_metadata_app_kwargs,
                                    "gradio_launch_options": gradio_launch_options})
                        if name == "app":
                            application.launch.assert_not_called()
                        else:
                            app_kwargs = application.launch.call_args.kwargs["app_kwargs"]
                            self.assertEqual(set(app_kwargs), {"middleware"})
                            self.assertEqual(len(app_kwargs["middleware"]), 1)
                            self.assertEqual(tuple(app_kwargs["middleware"][0]),
                                             tuple(social_metadata_app_kwargs()["middleware"][0]))
                            application.launch.assert_called_once_with(
                                favicon_path=str(ROOT / "assets/measured-rag-favicon-64.png"),
                                app_kwargs=app_kwargs,
                                **gradio_launch_options(mode, True))

    def test_configuration_and_build_integration(self):
        tree = ast.parse((ROOT / "config.py").read_text(encoding="utf-8-sig"))
        assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "DEPLOYMENT_MODE"
                               for t in node.targets)]
        self.assertEqual(len(assignments), 1)
        self.assertEqual(ast.unparse(assignments[0].value), "os.getenv('DEPLOYMENT_MODE', 'local')")
        docker = (ROOT / "Dockerfile").read_text()
        self.assertIn("DEPLOYMENT_MODE=cloud", docker)
        self.assertIn("COPY deployment/runtime.py /app/deployment/", docker)
        self.assertNotRegex(docker, r"\bPORT=")
        ignores = [(ROOT / name).read_text() for name in (".dockerignore", ".gcloudignore")]
        self.assertEqual(*ignores)
        self.assertIn("!deployment/runtime.py", ignores[0].splitlines())


if __name__ == "__main__":
    unittest.main()
