"""Explicit local and cloud Gradio launch configuration."""

import os
from pathlib import Path
from html import escape
from html.parser import HTMLParser

from starlette.middleware import Middleware


SOCIAL_TITLE = "Measured RAG \u2014 Document intelligence, measured end to end"
SOCIAL_DESCRIPTION = (
    "Measured RAG is a lightweight, open-source, production-style system for "
    "question answering across PDF and Word documents. Its interface exposes "
    "chunking and retrieval controls alongside hybrid dense and keyword retrieval, "
    "reranking, grounded answer generation, runtime telemetry, and reproducible "
    "benchmarking\u2014making the path from document ingestion to evaluated answer "
    "visible, configurable, and deployment-validated."
)
SOCIAL_URL = "https://demo.measured-rag.com/"
SOCIAL_IMAGE = SOCIAL_URL + "social-card-v3.png"
SOCIAL_CARD_PATH = Path(__file__).resolve().parents[1] / "assets" / "measured-rag-card-v3.png"
SOCIAL_CARD_CACHE_CONTROL = "public, max-age=31536000, immutable"
SOCIAL_OG = {
    "og:type": "website", "og:locale": "en_US", "og:site_name": "Measured RAG",
    "og:title": SOCIAL_TITLE, "og:description": SOCIAL_DESCRIPTION,
    "og:url": SOCIAL_URL, "og:image": SOCIAL_IMAGE,
    "og:image:url": SOCIAL_IMAGE, "og:image:secure_url": SOCIAL_IMAGE,
    "og:image:type": "image/png",
    "og:image:width": "1200", "og:image:height": "630",
    "og:image:alt": SOCIAL_TITLE,
}
SOCIAL_TWITTER = {
    "twitter:card": "summary_large_image", "twitter:title": SOCIAL_TITLE,
    "twitter:description": SOCIAL_DESCRIPTION, "twitter:image": SOCIAL_IMAGE,
    "twitter:image:alt": SOCIAL_TITLE,
}


def _social_tags():
    tags = [f"<title>{escape(SOCIAL_TITLE)}</title>",
            f'<meta name="description" content="{escape(SOCIAL_DESCRIPTION, quote=True)}">',
            f'<link rel="canonical" href="{escape(SOCIAL_URL, quote=True)}">',
            f'<link rel="image_src" href="{escape(SOCIAL_IMAGE, quote=True)}">']
    for attribute, fields in (("property", SOCIAL_OG), ("name", SOCIAL_TWITTER)):
        tags.extend(f'<meta {attribute}="{key}" content="{escape(value, quote=True)}">'
                    for key, value in fields.items())
    tags.extend(
        f'<meta itemprop="{key}" content="{escape(value, quote=True)}">'
        for key, value in (("name", SOCIAL_TITLE), ("description", SOCIAL_DESCRIPTION),
                           ("image", SOCIAL_IMAGE))
    )
    return "".join(tags)


class _PreviewTags(HTMLParser):
    """Locate preview tags without reserializing scripts, styles or other markup."""

    def __init__(self, html):
        super().__init__(convert_charrefs=False)
        self.html = html
        self.lines = [0]
        for line in html.split("\n"):
            self.lines.append(self.lines[-1] + len(line) + 1)
        self.in_head = False
        self.head_end = None
        self.title_start = None
        self.spans = []
        self.feed(html)
        self.close()

    def position(self):
        line, column = self.getpos()
        return self.lines[line - 1] + column

    def handle_starttag(self, tag, attrs):
        if tag == "head":
            self.in_head = True
        if not self.in_head:
            return
        attrs = dict(attrs)
        if tag == "title":
            self.title_start = self.position()
        elif self._is_preview_tag(tag, attrs):
            self.spans.append((self.position(), self.position() + len(self.get_starttag_text())))

    @staticmethod
    def _is_preview_tag(tag, attrs):
        if tag == "link":
            return bool({"canonical", "image_src"} & set((attrs.get("rel") or "").lower().split()))
        if tag != "meta":
            return False
        names = ((attrs.get(key) or "").lower() for key in ("name", "property"))
        if any(name == "description" or name.startswith(("og:", "twitter:")) for name in names):
            return True
        return (attrs.get("itemprop") or "").lower() in {"name", "description", "image"}

    def handle_endtag(self, tag):
        if tag == "title" and self.title_start is not None:
            self.spans.append((self.title_start, self.html.index(">", self.position()) + 1))
            self.title_start = None
        elif tag == "head" and self.in_head:
            self.head_end = self.position()
            self.in_head = False


def normalize_social_metadata(html):
    """Replace only head preview tags; leave all other source text untouched."""
    parsed = _PreviewTags(html)
    if parsed.head_end is None:
        return html
    edits = [(start, end, "") for start, end in parsed.spans]
    edits.append((parsed.head_end, parsed.head_end, _social_tags()))
    for start, end, replacement in sorted(edits, reverse=True):
        html = html[:start] + replacement + html[end:]
    return html


class SocialMetadataMiddleware:
    """Normalize the root HTML before Gradio's outer compression middleware."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path") == "/social-card-v3.png":
            await self._serve_social_card(scope, send)
            return
        if (scope["type"] != "http" or scope.get("path") != "/"
                or scope.get("method") not in {"GET", "HEAD"}):
            await self.app(scope, receive, send)
            return
        start = None
        chunks = []

        async def normalize_send(message):
            nonlocal start
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                if (headers.get(b"content-type", b"").split(b";", 1)[0].strip().lower()
                        == b"text/html" and headers.get(b"content-encoding", b"identity") == b"identity"
                        and message["status"] not in {204, 206, 304}):
                    start = message
                    return
            if message["type"] == "http.response.body" and start is not None:
                chunks.append(message.get("body", b""))
                if message.get("more_body", False):
                    return
                body = b"".join(chunks)
                try:
                    normalized = normalize_social_metadata(body.decode("utf-8")).encode("utf-8")
                except UnicodeDecodeError:
                    normalized = body
                if normalized != body:
                    stale = {b"content-length", b"etag", b"last-modified", b"content-md5",
                             b"digest", b"content-digest", b"repr-digest"}
                    headers = [(key, value) for key, value in start.get("headers", [])
                               if key.lower() not in stale]
                    headers.append((b"content-length", str(len(normalized)).encode("ascii")))
                    start = {**start, "headers": headers}
                await send(start)
                await send({**message, "body": normalized})
                return
            await send(message)

        await self.app(scope, receive, normalize_send)

    @staticmethod
    async def _serve_social_card(scope, send):
        """Serve only the versioned crawler card without involving Gradio's file API."""
        method = scope.get("method")
        if method not in {"GET", "HEAD"}:
            await send({"type": "http.response.start", "status": 405,
                        "headers": [(b"allow", b"GET, HEAD")]})
            await send({"type": "http.response.body", "body": b""})
            return
        body = SOCIAL_CARD_PATH.read_bytes()
        headers = [
            (b"content-type", b"image/png"),
            (b"content-length", str(len(body)).encode("ascii")),
            (b"content-disposition", b"inline"),
            (b"cache-control", SOCIAL_CARD_CACHE_CONTROL.encode("ascii")),
        ]
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        await send({"type": "http.response.body", "body": b"" if method == "HEAD" else body})


def social_metadata_app_kwargs():
    """FastAPI constructor options passed through Gradio's supported launch API."""
    return {"middleware": [Middleware(SocialMetadataMiddleware)]}


def gradio_launch_options(deployment_mode, share):
    """Keep local defaults; require the platform's ingress port in cloud mode."""
    if deployment_mode == "local":
        return {"share": share}
    if deployment_mode != "cloud":
        raise ValueError('DEPLOYMENT_MODE must be exactly "local" or "cloud".')

    raw_port = os.environ.get("PORT")
    if raw_port is None:
        raise ValueError("Cloud deployment requires the PORT environment variable.")
    try:
        port = int(raw_port)
    except ValueError:
        raise ValueError("Cloud PORT must be an integer between 1 and 65535.") from None
    if not 1 <= port <= 65535:
        raise ValueError("Cloud PORT must be between 1 and 65535.")
    return {"share": False, "server_name": "0.0.0.0", "server_port": port}
